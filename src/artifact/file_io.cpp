#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#    include "core/os.h"
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#if defined(_WIN32)

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " +
                        std::system_category().message(static_cast<int>(::GetLastError())));
}

HANDLE as_handle(std::intptr_t value) noexcept { return reinterpret_cast<HANDLE>(value); }

// Overlapped handles allow concurrent positional reads on one InputFile without a shared file
// pointer. Full sharing matches POSIX: an open artifact can still be rewritten or replaced. A
// failed open yields INVALID_HANDLE_VALUE, which is -1 like a failed POSIX open.
std::intptr_t open_handle(const std::filesystem::path& path, DWORD flags) {
    const HANDLE handle = ::CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_OVERLAPPED | flags, nullptr);
    return reinterpret_cast<std::intptr_t>(handle);
}

// One positional read of at most 1 GiB; returns the byte count (0 at EOF).
std::size_t read_at(HANDLE handle, std::uint64_t offset, std::byte* destination, std::size_t size,
                    const std::filesystem::path& path, const char* operation) {
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    overlapped.hEvent     = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (overlapped.hEvent == nullptr) { fail(path, operation); }
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
        if (error == ERROR_HANDLE_EOF) { return 0; }
        ::SetLastError(error);
        fail(path, operation);
    }
    return received;
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

#if defined(_WIN32)

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = open_handle(path_, FILE_FLAG_RANDOM_ACCESS);
    if (fd_ == -1) { fail(path_, "open"); }

    BY_HANDLE_FILE_INFORMATION info{};
    if (::GetFileInformationByHandle(as_handle(fd_), &info) == FALSE) {
        const DWORD error = ::GetLastError();
        ::CloseHandle(as_handle(fd_));
        fd_ = -1;
        ::SetLastError(error);
        fail(path_, "stat");
    }
    if ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
        ::GetFileType(as_handle(fd_)) != FILE_TYPE_DISK) {
        ::CloseHandle(as_handle(fd_));
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32U) | info.nFileSizeLow;
}

InputFile::~InputFile() {
    if (direct_fd_ != -1) { ::CloseHandle(as_handle(direct_fd_)); }
    if (fd_ != -1) { ::CloseHandle(as_handle(fd_)); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read =
            read_at(as_handle(fd_), offset, destination.data(), count, path_, "read");
        if (read == 0) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ == -1) {
        direct_fd_ = open_handle(path_, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN);
        if (direct_fd_ == -1) { fail(path_, "open direct"); }
    }
    // Unbuffered reads need sector-multiple sizes; the 1 GiB request cap is itself aligned. A short
    // final block at EOF ends the loop.
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto read = read_at(as_handle(direct_fd_), offset + total, destination.data() + total,
                                  destination.size() - total, path_, "direct read");
        if (read == 0) { break; }
        total += read;
        if (read % kPayloadAlignment != 0) { break; }
    }
    return total;
}

#else

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(static_cast<int>(fd_), &status) != 0) {
        const auto error = errno;
        ::close(static_cast<int>(fd_));
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(static_cast<int>(fd_));
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(static_cast<int>(direct_fd_)); }
    if (fd_ >= 0) { ::close(static_cast<int>(fd_)); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read =
            ::pread(static_cast<int>(fd_), destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(static_cast<int>(direct_fd_), destination.data(), destination.size(),
                       file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

#endif

} // namespace ninfer::artifact
