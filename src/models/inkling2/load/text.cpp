#include "models/inkling2/load/bindings.h"

namespace ninfer::models::inkling2::loading {

namespace {

constexpr QType kDirect = QType::FP32;  // converter stores norms/clips/biases/convs as fp32

} // namespace

AttentionLayerWeights bind_layer(Bindings& b, const Config& cfg, std::uint32_t index) {
    auto&& c_ = cfg; auto& c = c_;
    const auto p    = "text/layers/" + std::to_string(index) + "/";
    const auto h    = std::uint64_t(c.hidden_size);
    const auto q    = c.query_width();
    const auto k    = c.key_width();
    const auto& moe = cfg.moe;
    const auto mix  = p + "mixer_input";
    const auto ffn  = p + "ffn_input";

    AttentionLayerWeights w;
    const auto direct = [&](const std::string& name, artifact::Shape shape) {
        return b.binder.contains(name) ? b.direct(name, shape, kDirect) : WeightId{};
    };
    const auto matrix = [&](const std::string& name, artifact::Shape shape, std::string input) {
        // `name` is relative to the layer path; the contains() check must use the full id.
        return b.binder.contains(p + name)
                   ? b.parameter(p + name, shape, {std::move(input)})
                   : WeightId{};
    };

    w.input_norm       = direct(p + "input_norm", {h});
    // Full-attention layers store residual_query pre-tiled to [3][hidden] with three
    // DISTINCT mix rows (measured on weight/000170, 2026-11-25); sliding layers one row.
    const bool full = c.full_attention_layer.at(index);
    w.residual_query = full ? direct(p + "residual_query", {std::uint64_t(3), h})
                            : direct(p + "residual_query", {h});
    w.query            = matrix("attention/query", {q, h}, mix);
    w.key              = matrix("attention/key", {k, h}, mix);
    w.value            = matrix("attention/value", {k, h}, mix);
    w.gate             = matrix("attention/gate", {q, h}, mix);
    w.output           = matrix("attention/output", {h, q}, p + "attention/gated_output");
    w.query_input_norm = direct(p + "attention/query_input_norm", {h});
    w.key_input_norm   = direct(p + "attention/key_input_norm", {h});
    w.value_input_norm = direct(p + "attention/value_input_norm", {h});
    w.query_clip       = direct(p + "attention/query_clip", {c.num_attention_heads, c.head_dim});
    w.relative_clip    = direct(p + "attention/relative_clip",
                                {c.num_attention_heads, cfg.attention.relative_taps});
    w.relative_bias =
        direct(p + "attention/relative_bias", {k, cfg.attention.relative_taps});
    w.relative_projection =
        matrix("attention/relative_projection",
               {std::uint64_t(c.num_attention_heads) * cfg.attention.relative_taps, h}, mix);
    // v2 artifact: sink ships direct fp32 (executor gemm_f32 reads raw rows).
    w.sink_projection = direct(p + "attention/sink_projection", {c.num_attention_heads, h});
    w.key_convolution =
        direct(p + "attention/key_convolution", {k, c.conv_taps});
    w.value_convolution =
        direct(p + "attention/value_convolution", {k, c.conv_taps});
    w.post_convolution =
        direct(p + "attention/post_convolution", {h, c.conv_taps});
    // Measured: {32,2304} = kv_heads × conv_taps rows (per-KV-head tap mixing), full layers
    // only. v2 artifact stores these direct fp32 (executor gemm_f32 reads raw rows).
    const auto conv_proj_rows = c.key_width() / c.head_dim * c.conv_taps;
    w.key_convolution_projection =
        direct(p + "attention/key_convolution_projection", {conv_proj_rows, h});
    w.value_convolution_projection =
        direct(p + "attention/value_convolution_projection", {conv_proj_rows, h});

    w.post_attention_norm   = direct(p + "post_attention_norm", {h});
    w.ffn_residual_query    = direct(p + "ffn_residual_query", {h});
    w.ffn_post_convolution  = direct(p + "ffn_post_convolution", {h, c.conv_taps});

    w.moe.router      = matrix("moe/router", {moe.num_experts, h}, ffn);
    w.moe.router_bias = direct(p + "moe/router_bias", {moe.num_experts});
    w.moe.experts.reserve(moe.num_experts);
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "moe/experts/" + std::to_string(e) + "/";
        w.moe.experts.push_back(
            {b.parameter(ep + "gate", {moe.intermediate_size, h}, {ffn}),
             b.parameter(ep + "up", {moe.intermediate_size, h}, {ffn}),
             b.parameter(ep + "down", {h, moe.intermediate_size}, {ep + "product"})});
    }

    if (cfg.loop.loops(index)) {
        const auto rank = std::uint64_t(cfg.loop.controller_rank);
        const auto ctrl = p + "controller/";
        AttentionLayerWeights::Controller out;
        const auto cm = [&](const std::string& name, artifact::Shape shape, std::string input) {
            return b.parameter(ctrl + name, shape, {std::move(input)});
        };
        // Measured from the artifact `uses` table (2026-11-25): *_down reads the
        // [message; state] concatenation, which the artifact names controller_input
        // (q_down is {rank, 2h}); gains/forgets carry input controller_state;
        // message_query has NO use entry (direct GEMV on the block input).
        out.message_query = b.direct(ctrl + "message_query", {4, h}, kDirect);
        const std::string streams[] = {"q", "k", "v", "ffn"};
        WeightId AttentionLayerWeights::Controller::*const gains[]  = {&AttentionLayerWeights::Controller::q_gain,
                                                &AttentionLayerWeights::Controller::k_gain,
                                                &AttentionLayerWeights::Controller::v_gain,
                                                &AttentionLayerWeights::Controller::ffn_gain};
        WeightId AttentionLayerWeights::Controller::*const forgets[] = {&AttentionLayerWeights::Controller::q_forget,
                                                 &AttentionLayerWeights::Controller::k_forget,
                                                 &AttentionLayerWeights::Controller::v_forget,
                                                 &AttentionLayerWeights::Controller::ffn_forget};
        WeightId AttentionLayerWeights::Controller::*const downs[]  = {&AttentionLayerWeights::Controller::q_down,
                                                &AttentionLayerWeights::Controller::k_down,
                                                &AttentionLayerWeights::Controller::v_down,
                                                &AttentionLayerWeights::Controller::ffn_down};
        for (std::size_t s = 0; s < 4; ++s) {
            out.*gains[s]   = cm(streams[s] + "_gain", {h, rank}, p + "controller_state");
            out.*forgets[s] = cm(streams[s] + "_forget", {h, rank}, p + "controller_state");
            out.*downs[s]   = cm(streams[s] + "_down", {rank, 2 * h}, p + "controller_input");
        }
        w.controller = std::move(out);
    }
    return w;
}

TextWeights bind_text(Bindings& b, const Config& cfg) {
    const auto& c = cfg;
    const auto h  = std::uint64_t(c.hidden_size);
    TextWeights w;
    w.token_embedding     = b.parameter("text/token_embedding", {c.vocab_size, h});
    w.embedding_norm      = b.direct("text/embedding_norm", {h}, kDirect);
    w.final_norm          = b.direct("text/final_norm", {h}, kDirect);
    w.final_residual_query = b.direct("text/final_residual_query", {h}, kDirect);
    w.output_head         = b.parameter("text/output_head", {c.vocab_size, h},
                                {"text/final_hidden"});
    w.layers.reserve(c.num_hidden_layers);
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i)
        w.layers.push_back(bind_layer(b, cfg, i));
    return w;
}

} // namespace ninfer::models::inkling2::loading
