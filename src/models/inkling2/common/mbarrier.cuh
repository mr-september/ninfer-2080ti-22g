#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/pre_sm80.cuh"

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
#endif
}

// Every lane of a consumer warp calls this after its last read of a pipeline stage. It returns the
// stage to the producer whose TMA refills it. That refill is an async-proxy write. An arrive alone
// does not order it after the warp's generic-proxy reads (ldmatrix, ld.shared): ptxas issues the
// arrive while those reads are still pending, and the refill can then overwrite bytes that a lane
// has not read yet. The proxy fence orders the reads first.
__device__ __forceinline__ void cta_mbarrier_release_stage(std::uint64_t* barrier, int lane) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
    __syncwarp();
    if (lane == 0) cta_mbarrier_arrive(barrier);
#endif
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#endif
}

} // namespace ninfer::ops
