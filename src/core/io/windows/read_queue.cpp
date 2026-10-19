#include "core/io/read_queue.h"

#include "core/os.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace ninfer::io {
namespace {

constexpr ULONG_PTR kWakeKey = 1;

[[noreturn]] void fail(DWORD error, const char* operation) {
    throw std::system_error(static_cast<int>(error), std::system_category(), operation);
}

} // namespace

// Reads are issued with ReadFile on handles associated with one I/O completion port, which plays
// the role of the Linux ring: it is the single place completions and wake() are observed.
// ReadCompletion::error carries the Win32 error code on this platform.
struct DirectReadQueue::Impl {
    struct Slot {
        OVERLAPPED overlapped{}; // First member: the completion port returns its address.
        DirectRead request{};
        bool busy = false;
    };

    HANDLE port            = nullptr;
    bool closed            = false;
    std::uint32_t capacity = 0;
    std::size_t pending    = 0;
    std::atomic<bool> signalled{false};
    std::vector<FileView> files;
    std::vector<HANDLE> handles; // Per-queue handles: a file object has one completion port.
    std::vector<MemorySpan> buffers;
    std::vector<Slot> slots;          // Never resized: OVERLAPPED addresses stay stable.
    std::deque<ReadCompletion> ready; // Completions already taken from the port.

    ~Impl() {
        for (const HANDLE handle : handles)
            if (handle != nullptr && handle != INVALID_HANDLE_VALUE) ::CloseHandle(handle);
        if (port != nullptr) ::CloseHandle(port);
    }

    // Translates one finished read into a completion and frees its buffer.
    ReadCompletion retire(Slot& slot) {
        DWORD bytes = 0;
        DWORD error = ERROR_SUCCESS;
        if (::GetOverlappedResult(handles[slot.request.file_index], &slot.overlapped, &bytes,
                                  FALSE) == FALSE) {
            error = ::GetLastError();
            bytes = 0;
            if (error == ERROR_HANDLE_EOF) error = ERROR_SUCCESS; // A zero-byte read at EOF.
        }
        slot.busy = false;
        return {slot.request.tag, slot.request.buffer_index, static_cast<std::uint32_t>(bytes),
                static_cast<int>(error)};
    }

    // Takes up to `limit` entries off the port; `timeout` is in milliseconds.
    void collect(std::size_t limit, DWORD timeout) {
        std::array<OVERLAPPED_ENTRY, 64> entries;
        const ULONG count = static_cast<ULONG>(std::min(limit, entries.size()));
        ULONG removed     = 0;
        if (::GetQueuedCompletionStatusEx(port, entries.data(), count, &removed, timeout, FALSE) ==
            FALSE) {
            const DWORD error = ::GetLastError();
            if (error == WAIT_TIMEOUT) return;
            fail(error, "poll direct read completion");
        }
        for (ULONG i = 0; i < removed; ++i) {
            if (entries[i].lpOverlapped == nullptr) continue; // wake() packet
            auto* slot       = reinterpret_cast<Slot*>(entries[i].lpOverlapped);
            const auto index = static_cast<std::size_t>(slot - slots.data());
            if (index >= slots.size() || !slot->busy)
                throw std::logic_error("unmatched direct read completion");
            ready.push_back(retire(*slot));
        }
    }
};

DirectReadQueue::DirectReadQueue(std::span<const FileView> files,
                                 std::span<const MemorySpan> buffers, std::uint32_t capacity)
    : impl_(std::make_unique<Impl>()) {
    if (!capacity || capacity > buffers.size() || files.empty() ||
        files.size() > std::numeric_limits<int>::max() || buffers.size() > 65535)
        throw std::invalid_argument("invalid direct read queue capacity");
    auto& p    = *impl_;
    p.capacity = capacity;
    p.files.assign(files.begin(), files.end());
    p.buffers.assign(buffers.begin(), buffers.end());
    p.slots.resize(buffers.size());
    for (const auto file : files) {
        if (file.mode != ReadMode::Direct || file.descriptor < 0)
            throw std::invalid_argument("read queue requires direct files");
    }
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto buffer = buffers[i];
        validate_memory(buffer);
        if (buffer.device != Device::cpu() || !buffer.bytes ||
            buffer.bytes > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("read queue requires nonempty Host buffers");
        for (const auto file : files)
            validate_direct_read(file, 0, {static_cast<std::byte*>(buffer.data), buffer.bytes});
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[j].data);
            if (begin < other + buffers[j].bytes && other < begin + buffer.bytes)
                throw std::invalid_argument("read queue buffers overlap");
        }
    }
    p.port = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
    if (p.port == nullptr) fail(::GetLastError(), "create direct read completion port");
    for (const auto file : files) {
        // ReOpenFile yields a distinct file object: a file object has one completion port, and
        // the caller's handle stays untouched.
        const HANDLE source = reinterpret_cast<HANDLE>(file.descriptor);
        const HANDLE handle =
            ::ReOpenFile(source, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING);
        if (handle == INVALID_HANDLE_VALUE) fail(::GetLastError(), "reopen read file");
        p.handles.push_back(handle);
        if (::CreateIoCompletionPort(handle, p.port, 0, 1) == nullptr)
            fail(::GetLastError(), "associate read file with the completion port");
    }
}

DirectReadQueue::~DirectReadQueue() {
    try {
        close();
        drain();
    } catch (...) {
        // Releasing borrowed buffers with unknown kernel writers would violate ownership.
        std::terminate();
    }
}

ReadAdmission DirectReadQueue::try_enqueue(DirectRead request) {
    auto& p = *impl_;
    if (p.closed) return ReadAdmission::Closed;
    if (p.pending == p.capacity) return ReadAdmission::Full;
    if (request.file_index >= p.files.size() || request.buffer_index >= p.buffers.size() ||
        !request.bytes)
        throw std::invalid_argument("invalid direct read request");
    auto& slot        = p.slots[request.buffer_index];
    const auto buffer = p.buffers[request.buffer_index];
    if (slot.busy || request.bytes > buffer.bytes)
        throw std::invalid_argument("direct read buffer is busy or too small");
    validate_direct_read(p.files[request.file_index], request.file_offset,
                         {static_cast<std::byte*>(buffer.data), request.bytes});
    slot.overlapped            = {};
    slot.overlapped.Offset     = static_cast<DWORD>(request.file_offset & 0xFFFFFFFFULL);
    slot.overlapped.OffsetHigh = static_cast<DWORD>(request.file_offset >> 32U);
    slot.request               = request;
    // Issued here rather than in submit(): there is no submission queue to batch. A synchronous
    // success and ERROR_IO_PENDING both deliver a packet to the port.
    const BOOL ok = ::ReadFile(p.handles[request.file_index], buffer.data,
                               static_cast<DWORD>(request.bytes), nullptr, &slot.overlapped);
    ++p.pending;
    if (ok == FALSE && ::GetLastError() != ERROR_IO_PENDING) {
        // An issue failure is an ordinary failed completion; no packet will arrive for it.
        p.ready.push_back({request.tag, request.buffer_index, 0,
                           static_cast<int>(::GetLastError())});
        return ReadAdmission::Accepted;
    }
    slot.busy = true;
    return ReadAdmission::Accepted;
}

void DirectReadQueue::submit() {}

std::size_t DirectReadQueue::poll(std::span<ReadCompletion> output) {
    auto& p           = *impl_;
    std::size_t count = 0;
    while (count < output.size()) {
        if (p.ready.empty()) p.collect(output.size() - count, 0);
        if (p.ready.empty()) break;
        output[count++] = p.ready.front();
        p.ready.pop_front();
        --p.pending; // A request stays outstanding until poll delivers its completion.
    }
    return count;
}

ReadWake DirectReadQueue::wait() {
    auto& p = *impl_;
    for (;;) {
        if (p.signalled.exchange(false)) return ReadWake::Signalled;
        if (!p.ready.empty()) return ReadWake::Completion;
        p.collect(64, INFINITE);
        if (!p.ready.empty())
            return p.signalled.exchange(false) ? ReadWake::Signalled : ReadWake::Completion;
    }
}

void DirectReadQueue::wake() {
    impl_->signalled.store(true);
    if (::PostQueuedCompletionStatus(impl_->port, 0, kWakeKey, nullptr) == FALSE)
        fail(::GetLastError(), "wake direct read queue");
}

void DirectReadQueue::close() { impl_->closed = true; }

void DirectReadQueue::drain() {
    std::array<ReadCompletion, 128> completions;
    while (impl_->pending || !impl_->ready.empty())
        if (!poll(completions)) (void)wait();
}

std::size_t DirectReadQueue::outstanding() const noexcept { return impl_->pending; }

} // namespace ninfer::io
