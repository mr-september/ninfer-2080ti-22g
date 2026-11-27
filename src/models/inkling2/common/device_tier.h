#pragma once

// The architecture tier of the code a CUDA device runs in this binary.
//
// A build carries device code for one architecture or for several (CMAKE_CUDA_ARCHITECTURES
// "70;75;80;86;89" with CUDA 12, "75;80;86;89;120a" with CUDA 13). On each device CUDA loads the
// image of the newest of them that runs there. Device code selects its instructions from
// __CUDA_ARCH__ (NINFER_SM70_DEVICE, NINFER_SM75_DEVICE, NINFER_PRE_SM80_DEVICE); host code that
// sizes a launch, picks a kernel variant or a route by architecture asks for the device's tier, so
// host and device agree on every card of a pipeline whatever their mix.

#include <cstdint>

// Constexpr helpers that device code evaluates too.
#if defined(__CUDACC__)
#    define NINFER_TIER_HD __host__ __device__
#else
#    define NINFER_TIER_HD
#endif

namespace ninfer::ops {

enum class DeviceTier : std::uint8_t {
    // sm_70: the Ampere warp primitives emulated (sm70_emulation.cuh) and its own projection,
    // decode-attention and single-row prompt routes.
    Volta,
    // sm_75: Turing forms of the sm_80 instructions (sm75_mma.cuh) and kernels sized for 64 KiB of
    // shared memory per block.
    Turing,
    // sm_80, sm_86 and sm_89 code, which a 12.0 card also runs from a build without 120a.
    Ampere,
    // sm_120a code: the Ampere kernels plus Blackwell's FP8 and FP4 tensor-core routes.
    Blackwell,
};

// The tiers this binary carries code for (set by CMake from CMAKE_CUDA_ARCHITECTURES).
#if defined(NINFER_TIER_VOLTA)
inline constexpr bool kBuildVolta = true;
#else
inline constexpr bool kBuildVolta = false;
#endif
#if defined(NINFER_TIER_TURING)
inline constexpr bool kBuildTuring = true;
#else
inline constexpr bool kBuildTuring = false;
#endif
#if defined(NINFER_TIER_AMPERE)
inline constexpr bool kBuildAmpere = true;
#else
inline constexpr bool kBuildAmpere = false;
#endif
#if defined(NINFER_TIER_BLACKWELL)
inline constexpr bool kBuildBlackwell = true;
#else
inline constexpr bool kBuildBlackwell = false;
#endif
inline constexpr int kBuildTiers =
    int(kBuildVolta) + int(kBuildTuring) + int(kBuildAmpere) + int(kBuildBlackwell);
static_assert(kBuildTiers >= 1, "CMake defines the tier of every architecture it builds");
// Whether the binary carries a tier past Volta: the kernels Volta replaces with its own exist only
// then.
inline constexpr bool kBuildPostVolta = kBuildTuring || kBuildAmpere || kBuildBlackwell;

[[nodiscard]] constexpr bool tier_built(DeviceTier tier) {
    switch (tier) {
    case DeviceTier::Volta:
        return kBuildVolta;
    case DeviceTier::Turing:
        return kBuildTuring;
    case DeviceTier::Ampere:
        return kBuildAmpere;
    case DeviceTier::Blackwell:
        return kBuildBlackwell;
    }
    return false;
}

// The tier of the code CUDA runs on `device`, read once per device from the image a probe kernel
// loads there. A build of one tier answers without asking.
[[nodiscard]] DeviceTier device_tier(int device);
// The tier of the calling thread's current device.
[[nodiscard]] DeviceTier current_device_tier();

[[nodiscard]] inline bool current_device_volta() {
    if constexpr (!kBuildVolta) { return false; }
    return current_device_tier() == DeviceTier::Volta;
}

[[nodiscard]] inline bool current_device_turing() {
    if constexpr (!kBuildTuring) { return false; }
    return current_device_tier() == DeviceTier::Turing;
}

// Whether the current device runs the 120a code, and with it the FP8 and FP4 tensor-core routes.
[[nodiscard]] inline bool current_device_blackwell() {
    if constexpr (!kBuildBlackwell) { return false; }
    return current_device_tier() == DeviceTier::Blackwell;
}

// Kernel shapes come in two sets. Turing's are sized for its 64 KiB of shared memory per block;
// every other tier runs the standard set, sized for 99 KiB (sm_86, sm_89 and sm_120 allow that;
// sm_80 allows more, but its kernels are the sm_86 ones; a Volta card allows 96 KiB and runs the
// standard set through its emulated primitives, beside its own routes). A kernel whose shape
// depends on the set takes it as a template argument (TuringShapes), and its launcher picks the set
// of the current device (with_shape_set), so a card of a build of several architectures runs the
// shapes of its own build.
[[nodiscard]] NINFER_TIER_HD constexpr bool turing_shapes(DeviceTier tier) {
    return tier == DeviceTier::Turing;
}

inline constexpr bool kBuildTuringShapes   = kBuildTuring;
inline constexpr bool kBuildStandardShapes = kBuildVolta || kBuildAmpere || kBuildBlackwell;

// Shared memory one block of a shape set may opt into.
[[nodiscard]] NINFER_TIER_HD constexpr int shape_max_shared_bytes_per_block(bool turing_shapes) {
    return turing_shapes ? 64 * 1024 : 99 * 1024;
}

// Whether the image being compiled runs the shape set (true in the host pass). A build of several
// architectures instantiates each set's kernels in every image; a kernel whose set its image does
// not run returns at once there (`if constexpr (!shape_set_in_image(TuringShapes)) return;`), which
// no launch reaches, so that image carries no code for it.
[[nodiscard]] NINFER_TIER_HD constexpr bool shape_set_in_image(bool turing_shapes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
    return turing_shapes;
#elif defined(__CUDA_ARCH__)
    return !turing_shapes;
#else
    (void)turing_shapes;
    return true;
#endif
}

// Whether the calling thread's current device runs the Turing shape set.
[[nodiscard]] inline bool turing_shapes_here() { return current_device_turing(); }

// Calls `f.template operator()<TuringShapes>()` with the current device's shape set as a
// compile-time value, instantiating only the sets this binary carries.
template <class F>
decltype(auto) with_shape_set(F&& f) {
    if constexpr (kBuildTuringShapes && kBuildStandardShapes) {
        if (turing_shapes_here()) { return f.template operator()<true>(); }
        return f.template operator()<false>();
    } else {
        return f.template operator()<kBuildTuringShapes>();
    }
}

// Calls `f.template operator()<Tier>()` with the tier of the current device as a compile-time
// value, instantiating only the tiers this binary carries.
template <class F>
decltype(auto) with_current_tier(F&& f) {
    switch (current_device_tier()) {
    case DeviceTier::Volta:
        if constexpr (kBuildVolta) { return f.template operator()<DeviceTier::Volta>(); }
        break;
    case DeviceTier::Turing:
        if constexpr (kBuildTuring) { return f.template operator()<DeviceTier::Turing>(); }
        break;
    case DeviceTier::Ampere:
        if constexpr (kBuildAmpere) { return f.template operator()<DeviceTier::Ampere>(); }
        break;
    case DeviceTier::Blackwell:
        if constexpr (kBuildBlackwell) { return f.template operator()<DeviceTier::Blackwell>(); }
        break;
    }
#if defined(_MSC_VER) && !defined(__clang__)
    __assume(false);
#else
    __builtin_unreachable();
#endif
}

} // namespace ninfer::ops
