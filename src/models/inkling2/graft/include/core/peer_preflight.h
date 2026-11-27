#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer {

// What the host's PCI view says about a card before its peer mappings are trusted. Two
// configurations report peer capability (cudaDeviceCanAccessPeer == 1) and then lose or corrupt
// peer traffic without a CUDA error:
//
//   - an IOMMU that translates the card's DMA (group type "DMA" or "DMA-FQ"): peer requests fault
//     in the IOMMU; the peer path works with identity domains (`iommu=pt`), or with no IOMMU;
//   - a GeForce card under a driver with the P2P gate removed whose BAR1 does not cover its memory
//     (a VBIOS without resizable BAR): the driver falls back to mailbox mappings that point at
//     arbitrary memory, so stores vanish, loads return garbage, and a probe can reset the host.
//
// Absent facts (no sysfs, as on Windows, or no IOMMU group) are not refusals.
struct PeerDeviceFacts {
    std::string name;                       // cudaDeviceProp::name
    std::string pci_bus_id;                 // cudaDeviceGetPCIBusId spelling, e.g. 0000:3B:00.0
    std::uint64_t memory_bytes = 0;         // cudaDeviceProp::totalGlobalMem
    std::optional<std::string> iommu_type;  // iommu_group/type; nullopt: no group (no IOMMU)
    std::optional<std::uint64_t> bar1_bytes; // second line of the device's sysfs resource file
};

// Reads the IOMMU group type and BAR1 size of `pci_bus_id` under `sysfs_root` (normally "/sys").
// Missing files leave the fields unset.
[[nodiscard]] PeerDeviceFacts read_peer_device_facts(std::string_view sysfs_root,
                                                     std::string_view name,
                                                     std::string_view pci_bus_id,
                                                     std::uint64_t memory_bytes);

// Why peer access between two cards must not be used, or empty when the pair may be probed. Only
// a pair that passes is probed with real peer stores and loads.
[[nodiscard]] std::string peer_preflight_refusal(const PeerDeviceFacts& from,
                                                 const PeerDeviceFacts& to);

} // namespace ninfer
