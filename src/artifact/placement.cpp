#include "artifact/placement.h"

#include "artifact/framing.h"
#include "artifact/reader.h"

#include <algorithm>

namespace ninfer::artifact {

LayoutRegion plan_host_tensor_region(LayoutBuilder& layout, const Reader& reader,
                                     ObjectHandle handle) {
    const auto& geometry = reader.geometry(handle);
    const auto& tensor   = reader.directory().tensor(handle);
    if (geometry.bytes >= 2 * kPayloadAlignment) {
        const auto source    = reader.segments(tensor.offset, 1).front().file_offset;
        const auto bias      = source % kPayloadAlignment;
        const auto alignment = std::min<std::size_t>(geometry.alignment, kPayloadAlignment);
        if (bias % alignment == 0) {
            const auto cursor = layout.finish();
            auto offset       = layout.finish(geometry.alignment);
            offset += (bias + kPayloadAlignment - offset % kPayloadAlignment) % kPayloadAlignment;
            if (offset > cursor) (void)layout.add(offset - cursor, 1, "Host I/O alignment");
        }
    }
    return layout.add(geometry.bytes, geometry.alignment, tensor.id);
}

MaterializationPlan plan_materialization(const Reader& reader,
                                         std::span<const PlacementRequest> requests,
                                         PreparedResources resources) {
    if (resources.source && resources.source != &reader) {
        throw ArtifactError("prepared resources belong to another load session");
    }
    MaterializationPlan out;
    out.source       = &reader;
    out.object_count = reader.directory().objects.size();
    out.resources    = std::move(resources);
    std::vector<bool> host(out.object_count), registered(out.object_count),
        device(out.object_count), in_place(out.object_count);
    bool has_device = false;
    for (const auto& request : requests) {
        (void)reader.geometry(request.object);
        if (request.device.type() == DeviceType::CPU) {
            host.at(request.object.index) = true;
            if (request.host_access == HostAccess::CudaTransfer) {
                registered.at(request.object.index) = true;
            }
        } else {
            if (request.device.type() != DeviceType::CUDA ||
                (has_device && request.device != out.device) ||
                request.host_access == HostAccess::CudaTransfer) {
                throw ArtifactError("materialization requires one CUDA device");
            }
            out.device = request.device;
            has_device = true;
            if (request.host_access == HostAccess::CudaInPlace) {
                host.at(request.object.index)       = true;
                registered.at(request.object.index) = true;
                in_place.at(request.object.index)   = true;
            } else {
                device.at(request.object.index) = true;
            }
        }
    }
    LayoutBuilder host_layout, registered_layout, device_layout;
    for (std::size_t i = 0; i < out.object_count; ++i) {
        if (!host[i] && !device[i]) { continue; }
        const ObjectHandle handle{i};
        const auto& g    = reader.geometry(handle);
        const auto& name = reader.directory().tensor(handle).id;
        if (host[i]) {
            auto& layout = registered[i] ? registered_layout : host_layout;
            auto& placements =
                registered[i] ? out.registered_host_objects : out.ordinary_host_objects;
            placements.push_back({handle, plan_host_tensor_region(layout, reader, handle)});
            if (in_place[i]) { out.in_place_objects.push_back(handle); }
            out.host_alignment =
                std::max({out.host_alignment, static_cast<std::size_t>(g.alignment),
                          static_cast<std::size_t>(kPayloadAlignment)});
        }
        if (device[i]) {
            out.device_objects.push_back({handle, device_layout.add(g.bytes, g.alignment, name)});
        }
    }
    if (!out.registered_host_objects.empty()) {
        out.registered_host_bytes = registered_layout.finish(out.host_alignment);
    }
    LayoutBuilder host_storage;
    if (out.registered_host_bytes) {
        (void)host_storage.add(out.registered_host_bytes, out.host_alignment,
                               "registered Host weights");
    }
    if (host_layout.finish()) {
        (void)host_storage.add(host_layout.finish(), out.host_alignment, "ordinary Host weights");
    }
    out.host_capacity_bytes   = host_storage.finish();
    out.device_capacity_bytes = device_layout.finish();
    return out;
}

} // namespace ninfer::artifact
