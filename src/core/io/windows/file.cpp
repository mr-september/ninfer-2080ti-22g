#include "core/io/file.h"

#include "core/os.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::io {
namespace {

// Windows has no DIOALIGN query; 4 KiB covers 512-byte and 4 KiB sectors and matches the engine
// read unit.
constexpr std::size_t kDirectAlignment = 4096;

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       DWORD error = ::GetLastError()) {
    throw std::system_error(static_cast<int>(error), std::system_category(),
                            path.string() + ": " + operation);
}

HANDLE as_handle(std::intptr_t value) noexcept { return reinterpret_cast<HANDLE>(value); }

// Overlapped handles allow concurrent positional reads on one file without a shared file pointer.
// Full sharing matches POSIX: an open artifact can still be rewritten or replaced.
std::intptr_t open_handle(const std::filesystem::path& path, DWORD flags) {
    const HANDLE handle = ::CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_OVERLAPPED | flags, nullptr);
    return reinterpret_cast<std::intptr_t>(handle); // INVALID_HANDLE_VALUE is -1
}

// One positional read of at most 1 GiB; returns the byte count (0 at EOF).
std::size_t read_at(HANDLE handle, std::uint64_t offset, std::byte* destination, std::size_t size,
                    const std::filesystem::path& path, const char* operation) {
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    overlapped.hEvent     = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (overlapped.hEvent == nullptr) fail(path, operation);
    const DWORD request = static_cast<DWORD>(std::min<std::size_t>(size, 1ULL << 30U));
    DWORD received      = 0;
    BOOL ok             = ::ReadFile(handle, destination, request, nullptr, &overlapped);
    DWORD error         = ok != FALSE ? ERROR_SUCCESS : ::GetLastError();
    if (ok != FALSE || error == ERROR_IO_PENDING) {
        ok    = ::GetOverlappedResult(handle, &overlapped, &received, TRUE);
        error = ok != FALSE ? ERROR_SUCCESS : ::GetLastError();
    }
    ::CloseHandle(overlapped.hEvent);
    if (ok == FALSE) {
        if (error == ERROR_HANDLE_EOF) return 0;
        fail(path, operation, error);
    }
    return received;
}

} // namespace

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    view_.descriptor = open_handle(path, mode == ReadMode::Direct ? FILE_FLAG_NO_BUFFERING
                                                                  : FILE_FLAG_RANDOM_ACCESS);
    if (view_.descriptor == -1) fail(path, "open");
    view_.mode = mode;
    BY_HANDLE_FILE_INFORMATION info{};
    if (::GetFileInformationByHandle(as_handle(view_.descriptor), &info) == FALSE) {
        const DWORD error = ::GetLastError();
        ::CloseHandle(as_handle(view_.descriptor));
        view_.descriptor = -1;
        fail(path, "stat", error);
    }
    if ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
        ::GetFileType(as_handle(view_.descriptor)) != FILE_TYPE_DISK) {
        ::CloseHandle(as_handle(view_.descriptor));
        view_.descriptor = -1;
        fail(path, "expected regular file", ERROR_INVALID_PARAMETER);
    }
    view_.bytes = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32U) | info.nFileSizeLow;
    if (mode == ReadMode::Direct) {
        view_.memory_alignment = kDirectAlignment;
        view_.offset_alignment = kDirectAlignment;
    }
}

ReadOnlyFile::~ReadOnlyFile() {
    if (view_.descriptor != -1) ::CloseHandle(as_handle(view_.descriptor));
}

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (view_.descriptor != -1) ::CloseHandle(as_handle(view_.descriptor));
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor == -1)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail(path_, "read exceeds file length", ERROR_HANDLE_EOF);
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read =
            read_at(as_handle(view_.descriptor), offset, destination.data(), count, path_, "read");
        if (read == 0) fail(path_, "unexpected EOF", ERROR_HANDLE_EOF);
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    // Unbuffered reads need sector-multiple sizes; the 1 GiB request cap is itself aligned. A short
    // final block at EOF ends the loop.
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto read =
            read_at(as_handle(view_.descriptor), offset + total, destination.data() + total,
                    destination.size() - total, path_, "direct read");
        if (read == 0) break;
        total += read;
        if (read % view_.offset_alignment != 0) break;
    }
    return total;
}

} // namespace ninfer::io
