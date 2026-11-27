// The vector products (and Volta's matrix product) for one block type, in their own unit so they
// build in parallel with the others.
#include "ggml_bridge_vec.cuh"

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_IQ2_XXS);
NINFER_GGUF_SM70_MATRIX_INSTANCE(GGML_TYPE_IQ2_XXS);
