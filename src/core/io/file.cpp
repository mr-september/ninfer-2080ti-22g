#include "core/io/file.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::io {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       int error = errno) {
    throw std::system_error(error, std::generic_category(), path.string() + ": " + operation);
}

off_t position(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::invalid_argument("file offset exceeds off_t");
    return static_cast<off_t>(value);
}

} // namespace

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    view_.descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (mode == ReadMode::Direct ? O_DIRECT : 0));
    if (view_.descriptor < 0) fail(path, "open");
    view_.mode = mode;
    try {
        struct stat status {};

        if (::fstat(view_.descriptor, &status)) fail(path, "fstat");
        if (status.st_size < 0 || !S_ISREG(status.st_mode))
            fail(path, "expected regular file", EINVAL);
        view_.bytes = static_cast<std::uint64_t>(status.st_size);
        if (mode == ReadMode::Direct) {
            struct statx alignment {};

            const bool reported = ::statx(view_.descriptor, "", AT_EMPTY_PATH | AT_STATX_DONT_SYNC,
                                          STATX_DIOALIGN, &alignment) == 0 &&
                                  (alignment.stx_mask & STATX_DIOALIGN);
            view_.memory_alignment = reported ? alignment.stx_dio_mem_align : 4096;
            view_.offset_alignment = reported ? alignment.stx_dio_offset_align : 4096;
            if (!std::has_single_bit(view_.memory_alignment) ||
                !std::has_single_bit(view_.offset_alignment))
                fail(path, "unsupported direct I/O alignment", EOPNOTSUPP);
            if (!reported && view_.bytes) {
                // Local filesystems without DIOALIGN use the engine's 4 KiB read unit.
                std::unique_ptr<void, decltype(&std::free)> probe(std::aligned_alloc(4096, 4096),
                                                                  &std::free);
                if (!probe) throw std::bad_alloc();
                (void)read_direct(0, {static_cast<std::byte*>(probe.get()), 4096});
            }
        }
    } catch (...) {
        ::close(view_.descriptor);
        view_.descriptor = -1;
        throw;
    }
}

ReadOnlyFile::~ReadOnlyFile() {
    if (view_.descriptor >= 0) ::close(view_.descriptor);
}

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (view_.descriptor >= 0) ::close(view_.descriptor);
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor < 0)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail(path_, "read exceeds file length", ENODATA);
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto n     = ::pread(view_.descriptor, destination.data(), count, position(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            fail(path_, "pread");
        }
        if (!n) fail(path_, "unexpected EOF", ENODATA);
        offset += static_cast<std::uint64_t>(n);
        destination = destination.subspan(static_cast<std::size_t>(n));
    }
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    // A short read is legal (network and virtualized file systems cap each transfer); continue until
    // the request is filled or the file ends. Direct I/O keeps short reads block-aligned, so the
    // continuation stays aligned.
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto n = ::pread(view_.descriptor, destination.data() + total,
                               destination.size() - total, position(offset + total));
        if (n < 0) {
            if (errno == EINTR) continue;
            fail(path_, "direct pread");
        }
        if (n == 0) break;
        total += static_cast<std::size_t>(n);
        if (total % view_.offset_alignment != 0) break; // end of file inside a block
    }
    return total;
}

} // namespace ninfer::io
