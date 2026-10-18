// Windows has no GNU ld --wrap interposition, so the CUDA-call fault injection of
// materialization_cuda_errors.cpp is Linux/WSL-only. The rest of the materialization test still
// runs.
#include "artifact/binder.h"

namespace ninfer::test {

void materialization_cuda_errors(DeviceContext&) {}

} // namespace ninfer::test
