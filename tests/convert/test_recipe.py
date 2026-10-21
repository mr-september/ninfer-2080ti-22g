from __future__ import annotations

import struct
from dataclasses import replace

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.schema import binding_parts
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import grouped_absmax
from tools.convert.model import Model, Parameter
from tools.artifact.tensor_output import TensorOutput
from tools.convert.recipe import Recipe
from tools.convert.recipe import AuxiliaryReference
from tools.convert.methods import AuxiliaryValue, cast_direct
from tools.convert.sources.logical import array_source


def _model(names=("query", "key", "gate", "value")):
    model = Model({"text": {"config": {}}})
    for index, name in enumerate(names):
        values = (
            ((index + 1) * torch.tensor([1, 2, 4, 8], dtype=torch.bfloat16))[:, None]
            .expand(4, 128)
            .contiguous()
        )
        model.add(
            Parameter(name, (4, 128), array_source(values, name), inputs=("input",))
        )
    model.packing_groups = [
        group for group in (tuple(names), tuple(names[:2]), tuple(names[2:])) if group
    ]
    return model


def _write(path, model, prepared):
    specs = [job.spec for job in prepared.weights] + [
        spec for spec, _ in prepared.auxiliaries
    ]
    with ArtifactWriter(
        path,
        specs,
        components=model.components,
        bindings=prepared.bindings,
        uses=prepared.uses,
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
        for spec, data in prepared.auxiliaries:
            writer.write_object(spec.id, data)


def _assert_quantized_rows(artifact, part, amplitudes, format):
    object_id, begin, end = part
    obj = artifact.object(object_id)
    assert obj.format == format and end - begin == len(amplitudes) * 128
    scales, codes = decode_row_split_codes(
        artifact.read_object(object_id), format, obj.shape
    )
    first, last = begin // 128, end // 128
    maximum = {"q4_g64_fp16": 7, "q5_g64_fp16": 15}[format]
    assert bool((codes[first:last] == maximum).all())
    expected = [
        struct.unpack("<H", struct.pack("<e", value / maximum))[0]
        for value in amplitudes
    ]
    assert scales[first:last].view(torch.int16).tolist() == [
        [word, word] for word in expected
    ]


def test_expert_parents_preserve_bank_order_and_shared_auxiliary(tmp_path):
    model = Model({"text": {"config": {}}})
    values = {}
    for role, offset in (("gate", 0), ("up", 32)):
        data = (torch.arange(24) + offset).to(torch.bfloat16).reshape(3, 2, 4)
        values[role] = data
        model.add(
            Parameter(role, (3, 2, 4), array_source(data, role), inputs=("hidden",))
        )
    recipe = Recipe(model)
    recipe.auxiliary_tensor(
        "layer_scales", AuxiliaryValue("fp32", (1, 2), struct.pack("<ff", 2.0, 3.0))
    )
    for role in values:
        recipe.use(
            role,
            "hidden",
            auxiliaries={
                "activation_input_divisor": AuxiliaryReference("layer_scales", 0)
            },
        )
    # Deliberately insert reverse order; placement and Binding order are independent.
    for expert in (2, 1, 0):
        recipe.parent(
            [
                (role, expert * 8, (expert + 1) * 8, array_source(data[expert], role))
                for role, data in values.items()
            ],
            shape=(4, 4),
            format="bf16",
            method=cast_direct,
            order=(expert,),
        )
    prepared = recipe.prepare(device="cpu", rows_per_chunk=1)
    path = tmp_path / "experts.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        for role in values:
            actual = []
            parts = binding_parts(artifact.directory.bindings[role], artifact.by_id)
            assert len(parts) == 3
            for object_id, begin, end in parts:
                parent = torch.frombuffer(
                    bytearray(artifact.read_object(object_id)), dtype=torch.bfloat16
                )
                actual.append(parent[begin:end])
            assert torch.equal(torch.cat(actual), values[role].flatten())
        references = [
            use["auxiliaries"]["activation_input_divisor"]
            for use in artifact.directory.uses
        ]
        assert (
            references[0]
            == references[1]
            == {"parts": [{"object": "auxiliary/shared/layer_scales", "range": [0, 1]}]}
        )
        assert len(prepared.auxiliaries) == 1


def test_mixed_attention_uses_two_parents_and_distinct_permissions(tmp_path):
    model = _model()
    recipe = Recipe(model)
    recipe.assign(
        ("query", "key"),
        format="q4_g64_fp16",
        method=grouped_absmax,
        activation_policy="AllowA8",
    )
    recipe.assign(
        ("gate", "value"),
        format="q5_g64_fp16",
        method=grouped_absmax,
        activation_policy="A16Only",
    )
    prepared = recipe.prepare(device="cpu", rows_per_chunk=3)
    path = tmp_path / "mixed.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        parts = {
            name: binding_parts(binding, artifact.by_id)[0]
            for name, binding in artifact.directory.bindings.items()
        }
        assert parts["query"][0] == parts["key"][0]
        assert parts["gate"][0] == parts["value"][0]
        assert parts["query"][0] != parts["gate"][0]
        assert parts["query"][1:] == parts["gate"][1:] == (0, 512)
        assert parts["key"][1:] == parts["value"][1:] == (512, 1024)
        for index, name in enumerate(("query", "key", "gate", "value")):
            format = "q4_g64_fp16" if index < 2 else "q5_g64_fp16"
            _assert_quantized_rows(
                artifact,
                parts[name],
                [(index + 1) * row for row in (1, 2, 4, 8)],
                format,
            )
        assert {
            use["parameter"]: use["activation_policy"]
            for use in artifact.directory.uses
        } == {
            "query": "AllowA8",
            "key": "AllowA8",
            "gate": "A16Only",
            "value": "A16Only",
        }


def test_row_overrides_preserve_binding_order_and_coverage(tmp_path):
    model = _model(("weight",))
    model.packing_groups = []
    recipe = Recipe(model)
    recipe.assign("weight", format="q4_g64_fp16", method=grouped_absmax)
    recipe.assign("weight", rows=(1, 3), format="q5_g64_fp16")
    prepared = recipe.prepare(device="cpu", rows_per_chunk=1)
    path = tmp_path / "parts.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        parts = binding_parts(artifact.directory.bindings["weight"], artifact.by_id)
        assert len(parts) == 3
        for part, amplitudes, format in zip(
            parts, ([1], [2, 4], [8]), ("q4_g64_fp16", "q5_g64_fp16", "q4_g64_fp16")
        ):
            _assert_quantized_rows(artifact, part, amplitudes, format)


def test_shared_weight_keeps_use_independent_and_can_be_overridden():
    model = _model(("key", "context_key"))
    model.packing_groups = []
    recipe = Recipe(model)
    recipe.assign("*", format="q8_g32_fp16", method=grouped_absmax)
    recipe.share("context_key", "key")
    recipe.use("context_key", "input", activation_policy="AllowA8")
    shared = recipe.prepare(device="cpu")
    assert len(shared.weights) == 1
    assert shared.bindings["key"] == shared.bindings["context_key"]
    assert shared.uses[0]["activation_policy"] == "A16Only"
    assert shared.uses[1]["activation_policy"] == "AllowA8"
    recipe.assign("context_key", format="q5_g64_fp16")
    independent = recipe.prepare(device="cpu")
    assert len(independent.weights) == 2
    assert independent.bindings["key"] != independent.bindings["context_key"]


def test_private_component_storage_cannot_be_packed_with_target_weights():
    model = _model(("target", "draft"))
    model.packing_groups = []
    model.parameters["draft"] = replace(model.parameters["draft"], residency="dflash2")
    recipe = Recipe(model)
    recipe.group(("target", "draft"))
    with pytest.raises(ValueError, match="independently selected"):
        recipe.prepare(device="cpu")
    shared = Recipe(model)
    shared.share("draft", "target")
    prepared = shared.prepare(device="cpu")
    assert len(prepared.weights) == 1
    assert prepared.bindings["draft"] == prepared.bindings["target"]
