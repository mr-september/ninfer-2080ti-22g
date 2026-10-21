"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import cast_direct, grouped_absmax, import_encoded, stored_scale_mse
from .sources.fp8_block import fp8_block_source
from .sources.safetensors import tensor_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_8_flash_next(model, recipe, sources):
    """Qwen4Exp: Q8 projections, FP8 block MTP expert banks and paged NF4 PLE.

    The main routed experts are not converted: the previous recipe imported NVIDIA's NVFP4 banks, which the
    Turing product removed. The function raises after assigning everything else until the Qwen4 series
    defines their format.
    """
    recipe.provenance.update(
        mapping="qwen4_exp_v1",
        nvidia_mtp="preserve_e4m3fn_codes_bf16_block128x128_scales",
        stored_scale_mse="g16_fp64_exhaustive_e4m3fn_v1",
        q_gate_order="all_query_heads_then_all_gate_heads",
        convolution_order="tap_channel",
    )
    cfg = model.config
    expected = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": 2560,
        "vocab_size": 248320,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "ple_layer_ids": [1],
        "split_ngram_parts": 128,
        "ple_shard_shape": [2500012, 160],
        "ple_embed_dim": 2560,
        "layer_types": [
            "indexed_attention" if l % 4 == 3 else "linear_attention" for l in range(48)
        ],
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"qwen3_8_flash_next requires {key}={value!r}")
    recipe.place("text/*", (0,))
    for component in ("vision", "mtp"):
        if component in model.components:
            recipe.place(component + "/*", (1 if component == "vision" else 2, 0))
    for name in model.parameters:
        if name.startswith("text/layers/") and name.endswith(
            (
                "/gdn/query",
                "/gdn/key",
                "/gdn/value",
                "/gdn/z",
                "/gdn/output",
                "/attention/query",
                "/attention/output_gate",
                "/attention/output",
            )
        ):
            _assign(recipe, name, Q8)

    h, intermediate, experts = 2560, 640, 512
    width = h * intermediate
    nvidia = sources["nvidia"]

    if "mtp" in model.components:
        for role in ("token_embedding", "output_head"):
            recipe.share("mtp/" + role, "text/" + role)
        for name, parameter in model.parameters.items():
            if name.startswith("mtp/") and parameter.source_factory is not None:
                recipe.assign(name, source=model.source(name, nvidia))
        p = "mtp/layers/0/moe/experts/"
        gate_up_parts = []
        down_parts = []
        for expert in range(experts):
            refs = {}
            for role in ("gate", "up", "down"):
                refs[role] = fp8_block_source(
                    nvidia,
                    f"mtp.layers.0.mlp.experts.{expert}.{role}_proj",
                    (h, intermediate) if role == "down" else (intermediate, h),
                )
            first, last = expert * width, (expert + 1) * width
            gate_up_parts.extend(
                (p + role, first, last, refs[role]) for role in ("gate", "up")
            )
            down_parts.append((p + "down", first, last, refs["down"]))
        for part, shape, order in (
            (gate_up_parts, (experts, 2 * intermediate, h), 0),
            (down_parts, (experts, h, intermediate), 1),
        ):
            recipe.parent(
                part,
                shape=shape,
                format="fp8_e4m3fn_block128x128_bf16",
                method=import_encoded,
                order=(2, 1, order),
            )
        for role in ("gate", "up", "down"):
            name = p + role
            recipe.use(
                name,
                model.parameters[name].inputs[0],
                activation_policy="AllowA8",
            )

    rows, k = cfg["ple_shard_shape"]
    for shard in range(cfg["split_ngram_parts"]):
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        source = tensor_source(sources["base"], name, (rows, k))
        recipe.parent(
            [
                (
                    "text/layers/1/ple/table",
                    shard * rows * k,
                    (shard + 1) * rows * k,
                    source,
                )
            ],
            shape=(rows, k),
            format="nf4_g16_e4m3fn",
            method=stored_scale_mse,
            order=(4, shard),
        )

    raise NotImplementedError(
        "qwen3_8_flash_next: the main routed expert banks have no converter format on this fork "
        "(the NVIDIA NVFP4 import was removed); add them with the Qwen4 series recipe"
    )


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_8_flash_next": qwen3_8_flash_next,
}
