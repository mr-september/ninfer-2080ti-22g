"""Inkling-2-Mini (inkling2_loop GGUF) architecture adapter.

Maps the nokeepai Unbound GGUF tensor inventory and `inkling2_loop.*` metadata into
the repository's logical parameters. The GGUF stores tensors column-major; GGUFSource
already presents them in the repository's C-order (out, in) convention. The 24 stored
blocks run as 48 visits (prelude 0-5, core 6-17 x3, coda 18-23); the loop schedule is
represented in the component config, while bindings address the 24 stored parameter
sets once (shared weights are immutable, visit-local state is engine runtime state).

Source axis notes (verified against the tensor inventory in port/m0/geometry.json):
ffn_gate_up_exps per-expert rows are [gate 1152; up 1152]; conv tap tensors are stored
tap-fastest and become (channel, tap) after the axis flip; the four full-attention
blocks (sliding_window_pattern false: 5, 11, 17, 23) replace attn_norm with per-input
norms and add input-dependent K/V conv projections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .model import Model, Parameter
from .resources import load_resources
from .sources.gguf import GGUFSource
from .sources.logical import LogicalSource, select_rows

_ARCHITECTURE = "inkling2_loop"


def _positive(metadata: dict, key: str) -> int:
    value = metadata.get(f"{_ARCHITECTURE}.{key}")
    if type(value) is not int or value <= 0:
        raise ValueError(f"{_ARCHITECTURE}.{key}: expected a positive integer")
    return value


def _real(metadata: dict, key: str) -> float:
    value = metadata.get(f"{_ARCHITECTURE}.{key}")
    if type(value) not in (float, int):
        raise ValueError(f"{_ARCHITECTURE}.{key}: expected a real value")
    return float(value)


def _array(metadata: dict, key: str, length: int) -> list:
    value = metadata.get(f"{_ARCHITECTURE}.{key}")
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{_ARCHITECTURE}.{key}: expected {length} values")
    return list(value)


def text_config(metadata: dict) -> dict:
    if metadata.get("general.architecture") != _ARCHITECTURE:
        raise ValueError(
            f"expected general.architecture={_ARCHITECTURE!r}, "
            f"got {metadata.get('general.architecture')!r}"
        )
    layers = _positive(metadata, "block_count")
    heads = _array(metadata, "attention.head_count", layers)
    kv_heads = _array(metadata, "attention.head_count_kv", layers)
    windowed = _array(metadata, "attention.sliding_window_pattern", layers)
    if any(type(h) is not int or h <= 0 for h in heads) or any(
        type(h) is not int or h <= 0 for h in kv_heads
    ):
        raise ValueError("attention head counts must be positive integers")
    if len(set(heads)) != 1 or len(set(kv_heads)) != 1:
        raise ValueError("this port requires uniform head counts")
    if heads[0] % kv_heads[0]:
        raise ValueError("query heads must be divisible by KV heads")
    if any(value not in (True, False) for value in windowed):
        raise ValueError("sliding_window_pattern must be boolean per block")
    key_length = _positive(metadata, "attention.key_length")
    if _positive(metadata, "attention.value_length") != key_length:
        raise ValueError("this port requires equal key and value widths")
    if _positive(metadata, "attention.key_length_swa") != key_length:
        raise ValueError("this port requires one key width across tiers")
    if metadata.get(f"{_ARCHITECTURE}.rope.dimension_count", 0) != 0:
        raise ValueError("Inkling is no-RoPE; rope.dimension_count must be 0")
    start = _positive(metadata, "loop.start")
    end = _positive(metadata, "loop.end")
    if not start < end <= layers:
        raise ValueError("loop range must lie inside the stored blocks")
    if metadata.get(f"{_ARCHITECTURE}.loop.varying") not in (False, None):
        raise ValueError("only fixed (non-varying) loop bodies are implemented")
    return {
        "architectures": ["Inkling2LoopForCausalLM"],
        "model_type": "inkling2_loop_text",
        "hidden_size": _positive(metadata, "embedding_length"),
        "vocab_size": _positive(metadata, "vocab_size"),
        # The head is scored up to this row only (donor "cut to the unpadded vocabulary");
        # rows [unpadded, vocab) are padding and must never enter the softmax.
        "unpadded_vocab_size": _positive(metadata, "vocab_size_unpadded"),
        "num_hidden_layers": layers,
        "max_position_embeddings": _positive(metadata, "context_length"),
        "num_attention_heads": heads[0],
        "num_key_value_heads": kv_heads[0],
        "head_dim": key_length,
        "layer_types": [
            "sliding_attention" if value else "full_attention" for value in windowed
        ],
        "sliding_window": _positive(metadata, "attention.sliding_window"),
        "rms_norm_eps": _real(metadata, "attention.layer_norm_rms_epsilon"),
        "conv_taps": _positive(metadata, "shortconv.l_cache"),
        "num_experts": _positive(metadata, "expert_count"),
        "num_experts_per_tok": _positive(metadata, "expert_used_count"),
        "moe_intermediate_size": _positive(metadata, "expert_feed_forward_length"),
        "router_scale": _real(metadata, "expert_weights_scale"),
        "situ_beta": _real(metadata, "activation.situ_beta"),
        "situ_linear_beta": _real(metadata, "activation.situ_linear_beta"),
        "loop": {
            "start": start,
            "end": end,
            "count": _positive(metadata, "loop.count"),
            "controller_rank": _positive(metadata, "loop.gate_rank"),
        },
        "attn_res_block_size": _positive(metadata, "attn_res.block_size"),
        "final_logit_softcapping": _real(metadata, "final_logit_softcapping"),
        "attention": {
            "relative_taps": _positive(metadata, "attention.relative_length"),
            "relative_extent": _positive(metadata, "attention.relative_extent"),
            "qk_clip": _real(metadata, "attention.qk_clip"),
            "relative_clip": _real(metadata, "attention.relative_clip"),
            "log_scale_floor": _positive(metadata, "attention.log_scale_floor"),
            "log_scale_alpha": _real(metadata, "attention.log_scale_alpha"),
        },
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
    }


class _Builder:
    def __init__(self, model: Model, store: GGUFSource):
        self.model = model
        self.store = store

    def add(self, name, tensor, shape=None, *, inputs=(), direct="bf16", rows=None):
        source = self.store.tensor(tensor)
        if rows is not None:
            source = select_rows(source, rows)
        if shape is None:
            shape = source.shape  # row count carried from the stored tensor
        if tuple(source.shape) != tuple(shape):
            raise ValueError(f"{name}: source shape {source.shape} != {tuple(shape)}")
        self.model.add(
            Parameter(
                name,
                tuple(shape),
                source,
                None,
                tuple(inputs),
                direct,
                residency="text",
            )
        )

    def attention(self, prefix, block, config, full):
        h = config["hidden_size"]
        q = config["num_attention_heads"] * config["head_dim"]
        kv = config["num_key_value_heads"] * config["head_dim"]
        mixer = prefix + "mixer_input"
        for role, tensor, shape in (
            ("query", "attn_q", (q, h)),
            ("key", "attn_k", (kv, h)),
            ("value", "attn_v", (kv, h)),
            ("gate", "attn_gate", (q, h)),
        ):
            self.add(
                prefix + "attention/" + role,
                f"blk.{block}.{tensor}.weight",
                shape,
                inputs=(mixer,),
            )
        self.add(
            prefix + "attention/output",
            f"blk.{block}.attn_output.weight",
            (h, q),
            inputs=(prefix + "attention/gated_output",),
        )
        self.group(*(prefix + "attention/" + r for r in ("query", "key", "value", "gate")))
        heads = config["num_attention_heads"]
        taps = config["attention"]["relative_taps"]
        self.add(
            prefix + "attention/relative_projection",
            f"blk.{block}.attn_rel.weight",
            (taps * heads, h),
            inputs=(mixer,),
        )
        # Executor consumes sink/taps via gemm_f32 on raw fp32 rows (donor math), so
        # these small tensors ship direct, not quantized (measured 2026-11-25: the
        # q8 route broke the f32() resolve).
        self.add(
            prefix + "attention/sink_projection",
            f"blk.{block}.attn_sink_proj.weight",
            (heads, h),
            direct="fp32",
        )
        self.add(
            prefix + "attention/relative_bias",
            f"blk.{block}.attn_rel_bias.weight",
            (config["attention"]["relative_extent"], taps),
            direct="fp32",
        )
        self.add(
            prefix + "attention/relative_clip",
            f"blk.{block}.attn_rel_clip.weight",
            (heads, taps),
            direct="fp32",
        )
        self.add(
            prefix + "attention/query_clip",
            f"blk.{block}.attn_q_clip.weight",
            (heads, config["head_dim"]),
            direct="fp32",
        )
        for stream in ("key", "value"):
            self.add(
                prefix + f"attention/{stream}_convolution",
                f"blk.{block}.attn_{stream[0]}_conv.weight",
                (kv, config["conv_taps"]),
                direct="fp32",
            )
        self.add(
            prefix + "attention/post_convolution",
            f"blk.{block}.attn_post_conv.weight",
            (h, config["conv_taps"]),
            direct="fp32",
        )
        if full:
            for role in ("query", "key", "value"):
                self.add(
                    prefix + f"attention/{role}_input_norm",
                    f"blk.{block}.attn_{role[0]}_inp_norm.weight",
                    (h,),
                    direct="fp32",
                )
            for role in ("key", "value"):
                # Input-dependent taps are per KV head per tap: 8 * 4 rows.
                self.add(
                    prefix + f"attention/{role}_convolution_projection",
                    f"blk.{block}.attn_{role[0]}_conv_proj.weight",
                    (config["num_key_value_heads"] * config["conv_taps"], h),
                    direct="fp32",
                )
            self.add(
                prefix + "residual_query",
                f"blk.{block}.attn_res_query.weight",
                direct="fp32",
            )
        else:
            self.add(
                prefix + "input_norm",
                f"blk.{block}.attn_norm.weight",
                (h,),
                direct="fp32",
            )
            if block > 0:  # block 0 has no prior attention stream to query
                self.add(
                    prefix + "residual_query",
                    f"blk.{block}.attn_res_query.weight",
                    (h,),
                    direct="fp32",
                )

    def moe(self, prefix, block, config):
        h = config["hidden_size"]
        experts = config["num_experts"]
        intermediate = config["moe_intermediate_size"]
        ffn = prefix + "ffn_input"
        self.add(
            prefix + "moe/router",
            f"blk.{block}.ffn_gate_inp.weight",
            (experts, h),
            inputs=(ffn,),
            direct="fp32",
        )
        self.add(
            prefix + "moe/router_bias",
            f"blk.{block}.exp_probs_b.bias",
            (experts,),
            direct="fp32",
        )
        gate_up = self.store.tensor(f"blk.{block}.ffn_gate_up_exps.weight")
        gate_up_rows = LogicalSource(
            (experts * 2 * intermediate, h), gate_up.label, gate_up.read_values
        )
        down = self.store.tensor(f"blk.{block}.ffn_down_exps.weight")
        down_rows = LogicalSource(
            (experts * h, intermediate), down.label, down.read_values
        )
        gate_up_names, down_names = [], []
        for expert in range(experts):
            ep = prefix + f"moe/experts/{expert}/"
            base = expert * 2 * intermediate
            for role, offset in (("gate", 0), ("up", intermediate)):
                self.model.add(
                    Parameter(
                        ep + role,
                        (intermediate, h),
                        select_rows(
                            gate_up_rows, ((base + offset, base + offset + intermediate),)
                        ),
                        None,
                        (ffn,),
                        "bf16",
                        residency="text",
                    )
                )
                gate_up_names.append(ep + role)
            self.model.add(
                Parameter(
                    ep + "down",
                    (h, intermediate),
                    select_rows(down_rows, ((expert * h, (expert + 1) * h),)),
                    None,
                    (ep + "product",),
                    "bf16",
                    residency="text",
                )
            )
            down_names.append(ep + "down")
        self.group(*gate_up_names)
        self.group(*down_names)

    def controllers(self, prefix, block, config):
        # Rank-64 low-rank controller: *_down carries the hidden pair (2h) into the
        # state; *_forget/*_gain modulate the hidden streams from the state. Row
        # counts are carried from the stored tensors (axis order checked in M0).
        state = prefix + "controller_state"
        self.add(
            prefix + "controller/message_query",
            f"blk.{block}.loop_msg_query.weight",
            direct="fp32",
        )
        for stream in ("q", "k", "v", "ffn"):
            self.add(
                prefix + f"controller/{stream}_down",
                f"blk.{block}.loop_{stream}_ctrl_down.weight",
                inputs=(prefix + "controller_input",),
                direct="fp32",
            )
            for role in ("forget", "gain"):
                self.add(
                    prefix + f"controller/{stream}_{role}",
                    f"blk.{block}.loop_{stream}_ctrl_{role}.weight",
                    inputs=(state,),
                    direct="fp32",
                )

    def block(self, prefix, block, config):
        h = config["hidden_size"]
        full = config["layer_types"][block] == "full_attention"
        self.attention(prefix, block, config, full)
        self.add(
            prefix + "post_attention_norm",
            f"blk.{block}.ffn_norm.weight",
            (h,),
            direct="fp32",
        )
        self.add(
            prefix + "ffn_post_convolution",
            f"blk.{block}.ffn_post_conv.weight",
            (h, config["conv_taps"]),
            direct="fp32",
        )
        self.add(
            prefix + "ffn_residual_query",
            f"blk.{block}.ffn_res_query.weight",
            (h,),
            direct="fp32",
        )
        self.moe(prefix, block, config)
        start, end = config["loop"]["start"], config["loop"]["end"]
        if start <= block < end:
            self.controllers(prefix, block, config)

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))


def build_model(
    store: GGUFSource,
    *,
    resource_dir: str | Path,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    config = text_config(store.metadata)
    refs, resources, count, special = load_resources(
        Path(resource_dir),
        vocab_size=config["vocab_size"],
        overrides=resource_overrides,
    )
    model = Model(
        {"text": {"config": config, "resources": refs["text"]}},
        resources=resources,
        token_count=count,
        special_token_ids=special,
    )
    builder = _Builder(model, store)
    h = config["hidden_size"]
    builder.add(
        "text/token_embedding", "token_embd.weight", (config["vocab_size"], h)
    )
    builder.add(
        "text/embedding_norm", "token_embd_norm.weight", (h,), direct="fp32"
    )
    builder.add(
        "text/output_head",
        "output.weight",
        (config["vocab_size"], h),
        inputs=("text/final_hidden",),
    )
    builder.add("text/final_norm", "output_norm.weight", (h,), direct="fp32")
    builder.add(
        "text/final_residual_query", "output_res_query.weight", (h,), direct="fp32"
    )
    for block in range(config["num_hidden_layers"]):
        builder.block(f"text/layers/{block}/", block, config)
    return model
