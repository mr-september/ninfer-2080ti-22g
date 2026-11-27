// IQ1_M's routed-expert matrix product, in its own unit so the heavy template builds in parallel
// with the others. Its dense products keep the BF16 GEMM (has_matrix_kernel).
#include "ggml_bridge_mmq.cuh"

NINFER_GGUF_MOE_MATRIX_INSTANCE(GGML_TYPE_IQ1_M);
