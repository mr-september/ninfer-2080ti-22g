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

add_executable(ninfer-logit-compare logit_compare/main.cpp)
ninfer_internal_includes(ninfer-logit-compare)
target_link_libraries(ninfer-logit-compare PRIVATE
  ninfer_engine ninfer_product_prompt_input ninfer::json)
ninfer_windows_utf8(ninfer-logit-compare)
