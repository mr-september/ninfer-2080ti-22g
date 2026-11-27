# Basalt Inkling-2-Mini — kernel inventory for sm_75 port

Source: `C:/Users/Larry/AppData/Local/hermes/cache/scratch/inkling_port/basalt` (MIT, jesdga95/basalt, depth-1 clone).
Model geometry (from docs/models/inkling2.md + include/basalt/models/inkling2/kernels.hpp):
24 physical layers (prelude 0-5, core 6-17 x3 passes, coda 18-23); width 2304; 24 query heads, 8 KV heads x 128 head dim (3 query heads per KV head); no rotary; short conv 4 taps (kConv=4), ring 16 positions; relative bias 16 states (kRel=16), window 1024, full attention on layers 5/11/17/23; MoE 128 experts top-8 sigmoid, expert FFN 1152 wide, softcap SwiGLU; 8 full-attention visits/token; DSpark drafter (BF16); vision ViT 23-layer (BF16, separate).
General: decode kernels are fp32/fp16/e4m3; prompt kernels use NVFP4/Q8_0/MMQ. cp.async.bulk.prefetch.L2 and clusters appear in decode kernels (sm_90 features).

## Decode kernels (kernels/ops.cu, glue.cu, attn.cu, prep.cuh)

| kernel/function | what it computes | geometry | arch features | sm_75 difficulty |
|---|---|---|---|---|
| attnres_mix_kernel (ops.cu) | block-residual softmax mix over finished blocks (softmax over 6 blocks + current), fp32 | kWidth=2304, <=kMaxMix blocks, 192 threads | plain fp32 + warp shfl | easy |
| ctrl_pair_norm_kernel (ops.cu) | pair norm: rms(dest), rms(msg) unweighted, eps 1e-6, [2*kWidth] | 256 threads, kWidth=2304 | plain | easy |
| ctrl_blend_kernel (ops.cu) | (1-tanh(f))*dest + tanh(g)*msg | kWidth=2304 | plain | easy |
| attn_prepare_kernel (ops.cu) | conv inputs, clip keys, norm values, query clip, relative states, sinks (shared prep.cuh) | kHeads=24, kKvHeads=8, kHeadDim=128, kRel=16, kConv=4, kConvRing=16 | plain + fp8 unpack | easy |
| ring_store_kernel (ops.cu) | store conv ring rows | kConvRing=16 x 1024 | plain | easy |
| attn_gate_kernel (ops.cu) | output gate sigmoid(gate) | kHeads=24 x kHeadDim=128 | plain | easy |
| post_conv_kernel (ops.cu) | post-attention short conv over kWidth channels, inv = rsqrt(kConv+1) | kWidth=2304, 32-row tiles | plain | easy |
| moe_plan_kernel (ops.cu) | expert plan: counts, firsts, sids for router ids | n_expert=128, top-8 | plain | easy |
| moe_cpu_merge_kernel (ops.cu) | merge CPU-computed expert rows | k=8, n_expert=128 | plain | easy |
| expert_read_ahead_kernel (ops.cu) | read-ahead of expert blobs (warp 32) | 128 experts | cp.async.bulk.prefetch.L2 (sm_90) | medium — replace prefetch with __ldg / plain load |
| expert_prefetch_kernel (ops.cu) | prefetch expert weights into L2 | 128 experts | cp.async.bulk.prefetch.L2 (sm_90) | medium — same |
| moe_combine_kernel (ops.cu) | combine top-k expert outputs weighted | k=8, kWidth=2304 | plain | easy |
| softcap_kernel (ops.cu) | softcap: cap*tanh(x/cap) over x | kWidth=2304 | plain | easy |
| mix_norm_kernel (glue.cu) | mix + rms norm + q8_1 encode | kWidth=2304, kPer=3, 768 threads | plain + q8_1 encode | easy |
| ctrl_down_kernel (glue.cu) | loop controller down projection (BF16) | kWidth=2304, 8 warps x 576 inputs | plain fp32 | easy |
| ctrl_up_kernel (glue.cu) | loop controller up projection (BF16) | kWidth=2304, 256 threads | plain | easy |
| ctrl_blend_kernel (glue.cu) | controller blend (same as ops.cu but glue variant) | kWidth=2304 | plain | easy |
| prepare_kernel (glue.cu) | decode attention prep (shared prep.cuh) | 128 threads, kHeads=24, kHeadDim=128, kRel=16 | plain + fp8 | easy |
| post_attn_kernel (glue.cu) | post-attention norm/conv + L2 prefetch of rings | kWidth=2304, kConvRing=16 | cp.async.bulk.prefetch.L2 (sm_90) | medium — replace prefetch |
| moe_finish_kernel (glue.cu) | expert combine/finish + L2 prefetch | kWidth=2304, 256 threads | cp.async.bulk.prefetch.L2 (sm_90) | medium — replace prefetch |
| attn_split_kernel (attn.cu) | attention slice over key range, fp32 partials (ml, l) | kHeads=24, kKvHeads=8, kHeadDim=128, kTile=32, kWarps=4 | plain fp32 + shfl | easy |
| attn_combine_kernel (attn.cu) | merge split partials (softmax combine) | kHeadDim=128 | plain | easy |
| attn_fused_kernel (attn.cu) | fused prep+slices+merge in ONE kernel, cluster of 8 splits | kClusterSplits=8, kFusedRows=3, kWarps=4 | __cluster_dims__ + cluster shared-memory map + fence.release/acquire (sm_90) | hard — needs cluster replacement (e.g., plain multi-block + global sync) |

## Prompt kernels (kernels/attn_prompt.cu, prompt_ops.cu)

| kernel/function | what it computes | geometry | arch features | sm_75 difficulty |
|---|---|---|---|---|
| attn_prompt_kernel (attn_prompt.cu) | prompt attention over 64-position blocks, fp16 mma | kPos=64, kWarps=3*4=12, kKeys=32, kHeadDim=128, kRel=16, kBand=96 | ldmatrix (sm_75+), mma.m16n8k16 f16 (sm_75+), cp.async (sm_80+) | medium — cp.async needs sm_80 replacement (plain ld/st) |
| sort_count/scan/scatter_kernel (prompt_ops.cu) | expert sort by expert id (counts, scan, scatter) | k=8, n_expert=128 | plain | easy |
| stage_to_ring_kernel (prompt_ops.cu) | stage KV into ring | kHeadDim=128 | plain | easy |
| prompt_rows_kernel (prompt_ops.cu) | compute positions + bounds of prompt rows | rows | plain | easy |
| remap_ids_kernel (prompt_ops.cu) | remap expert ids through map | n | plain | easy |
| ctrl_down_prompt_kernel (prompt_ops.cu) | controller down projection (prompt) | kWidth=2304, kCdW warps | cp.async (sm_80+) | medium — replace cp.async |
| ctrl_up_blend_prompt_kernel (prompt_ops.cu) | controller up + blend (prompt) | kWidth=2304, kCuW warps | cp.async (sm_80+), mma.m16n8k8 tf32 (sm_80+) | medium — tf32 mma needs sm_80; replace with f16 mma or plain FMA |
| rms_norm_q8_1_prompt_kernel (prompt_ops.cu) | rms norm + q8_1 encode (prompt) | kWidth=2304 | plain | easy |
| attn_gate_q8_1_prompt_kernel (prompt_ops.cu) | attention gate + q8_1 encode (prompt) | kHeads*kHeadDim=3072 | plain | easy |
| moe_combine_f16_kernel (prompt_ops.cu) | moe combine in fp16 (prompt) | kWidth=2304 | plain | easy |

## DSpark drafter kernels (kernels/dspark.cu)

| kernel/function | what it computes | geometry | arch features | sm_75 difficulty |
|---|---|---|---|---|
| mix_kernel (dspark.cu) | DSpark mix over taps | kWidth=2304, kThreads=256 | plain fp32 | easy |
| block_inputs_kernel (dspark.cu) | block inputs (embedding gather) | kDsRank | plain | easy |
| markov_step_kernel (dspark.cu) | Markov step (ctx_h, ctx_f) | kDsRank, kStepWarps=32 | plain fp32 | easy |
| screen_pack_kernel (dspark.cu) | pack q8 weights for screen | kWidth=2304 | plain | easy |
| head_base_kernel (dspark.cu) | head base (q8 dot) | kWidth=2304, kHeadNb=72 | plain fp32 | easy |
| head_step_kernel (dspark.cu) | head step (screen + fold) | kDsRank, kTilesPerBlock=4 | plain fp32 | easy |
| head_final_kernel (dspark.cu) | head final (exact logit) | kDsRank, kW1Words | plain fp32 | easy |
| screen_rows_kernel (dspark.cu) | screen rows by token | tiles | plain | easy |
| head_exact_kernel (dspark.cu) | exact head | kDsRank | plain fp32 | easy |
| fold_kernel (dspark.cu) | fold head | kDsRank | plain | easy |
| fold_b_kernel (dspark.cu) | fold b | kDsRank | plain | easy |
| split_rnn_kernel (dspark.cu) | split RNN | kDsRank | plain | easy |

## Host-side (non-CUDA) files

| file | role |
|---|---|
| walk.cpp | row-buffer walk for the 48-layer-visit loop (prelude/loop/coda), buffer assignment, split_walk for 2-card split |
| spill.cpp | spill plan: which experts go to RAM (coldest-first by profile), staging/device bytes |
| dspark.cpp | DSpark drafter orchestration (801 lines) |
| session.cpp | decode session (1343 lines) |
| prompt.cpp | prompt processing (656 lines) |
| model.cpp | model forward (834 lines) |
| pack.cpp | pack loading (165 lines) |
| pretok_inkling2.cpp | pretokenization (485 lines) |
| cpu_experts.cpp | CPU expert computation (144 lines) |
| vision_encoder.cpp | ViT encoder (109 lines) |
| adapter.cpp | model adapter (85 lines) |

## sm_75 port summary
- Easiest: all plain fp32/fp16 kernels (majority).
- Medium: kernels using cp.async (sm_80+) — attn_prompt_kernel, ctrl_down/up_prompt_kernel; kernels using cp.async.bulk.prefetch.L2 (sm_90) — expert_read_ahead, expert_prefetch, post_attn, moe_finish; attn_prompt_kernel needs cp.async replacement.
- Hard: attn_fused_kernel (sm_90 clusters) — needs cluster-free rewrite; attn_prompt_kernel needs cp.async replacement (sm_80).
- ldmatrix (sm_75+) and mma.m16n8k16 f16 (sm_75+) are OK on sm_75. mma.m16n8k8 tf32 (sm_80+) in prompt_ops.cu needs replacement (f16 mma or plain FMA).
- No __CUDA_ARCH__ guards found; kernels assume sm_75+ for ldmatrix/mma and sm_80/90 for cp.async/clusters.
