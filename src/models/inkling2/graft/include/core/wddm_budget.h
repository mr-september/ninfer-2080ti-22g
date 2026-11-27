#pragma once

// A process-wide switch of how owning device arenas commit their memory on Windows (WDDM).

namespace ninfer::core {

// The WDDM evictable budget (Windows builds, --wddm-evictable-budget): every owning DeviceArena
// commits its pages when it is created, so WDDM makes room for them at startup rather than at the
// first launch that touches them. Other builds refuse to enable it.
void set_wddm_evictable_budget(bool enabled);
[[nodiscard]] bool wddm_evictable_budget() noexcept;

} // namespace ninfer::core
