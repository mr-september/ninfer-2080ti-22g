add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

ninfer_windows_utf8(ninfer)
ninfer_windows_utf8(ninfer-serve)
ninfer_windows_utf8(ninfer-perplexity)

add_executable(ninfer-inkling2-probe logit_compare/inkling2_probe.cpp)
ninfer_internal_includes(ninfer-inkling2-probe)
target_link_libraries(ninfer-inkling2-probe PRIVATE ninfer_engine ninfer_product_logging ninfer::json CUDA::cudart)
ninfer_windows_utf8(ninfer-inkling2-probe)

add_executable(ninfer-logit-compare logit_compare/main.cpp)
ninfer_internal_includes(ninfer-logit-compare)
target_link_libraries(ninfer-logit-compare PRIVATE
  ninfer_engine ninfer_product_prompt_input ninfer::json)
ninfer_windows_utf8(ninfer-logit-compare)

# Graft link-stage probe: exercises the graft lib against the port engine libs
# via /WHOLEARCHIVE so LNK output enumerates every missing donor implementation.
add_executable(ninfer-inkling2-graft-probe logit_compare/inkling2_graft_probe.cpp)
ninfer_internal_includes(ninfer-inkling2-graft-probe)
target_link_libraries(ninfer-inkling2-graft-probe PRIVATE
  ninfer_engine ninfer_inkling2_graft ninfer_inkling2_ggml ninfer_product_logging ninfer::json CUDA::cudart CUDA::cublasLt)
target_link_options(ninfer-inkling2-graft-probe PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:ninfer_inkling2_graft>)
