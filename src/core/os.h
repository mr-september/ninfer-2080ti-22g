#pragma once

// Small operating-system shim shared by the CLI, server and product libraries. Linux (including
// WSL2) and native Windows builds use the same sources; this header is the only place that names
// the OS API for process, time and console facts.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <initializer_list>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <fcntl.h>
#    include <io.h>
#else
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

namespace ninfer::os {

[[nodiscard]] inline long long process_id() noexcept {
#if defined(_WIN32)
    return static_cast<long long>(::GetCurrentProcessId());
#else
    return static_cast<long long>(::getpid());
#endif
}

// Thread-safe broken-down local / UTC time; false on failure.
[[nodiscard]] inline bool local_time(std::time_t seconds, std::tm& out) noexcept {
#if defined(_WIN32)
    return ::localtime_s(&out, &seconds) == 0;
#else
    return ::localtime_r(&seconds, &out) != nullptr;
#endif
}

[[nodiscard]] inline bool utc_time(std::time_t seconds, std::tm& out) noexcept {
#if defined(_WIN32)
    return ::gmtime_s(&out, &seconds) == 0;
#else
    return ::gmtime_r(&seconds, &out) != nullptr;
#endif
}

[[nodiscard]] inline bool stderr_is_terminal() noexcept {
#if defined(_WIN32)
    return ::_isatty(::_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) == 1;
#endif
}

[[nodiscard]] inline std::size_t stderr_columns(std::size_t fallback) noexcept {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (::GetConsoleScreenBufferInfo(::GetStdHandle(STD_ERROR_HANDLE), &info) != 0) {
        const int width = info.srWindow.Right - info.srWindow.Left + 1;
        if (width > 0) { return static_cast<std::size_t>(width); }
    }
    return fallback;
#else
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) { return size.ws_col; }
    return fallback;
#endif
}

// Process-wide console setup for the executables. On Windows: UTF-8 console code pages, ANSI escape
// processing on stdout/stderr, and untranslated "\n" on both streams (as on Linux). No-op
// elsewhere.
inline void init_console() noexcept {
#if defined(_WIN32)
    ::SetConsoleOutputCP(CP_UTF8);
    ::SetConsoleCP(CP_UTF8);
    for (const DWORD which : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
        const HANDLE handle = ::GetStdHandle(which);
        DWORD mode          = 0;
        if (handle != INVALID_HANDLE_VALUE && ::GetConsoleMode(handle, &mode) != 0) {
            (void)::SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
    (void)::_setmode(::_fileno(stdout), _O_BINARY);
    (void)::_setmode(::_fileno(stderr), _O_BINARY);
#endif
}

// Physical memory the OS can hand out now, in bytes; 0 when it cannot be determined.
[[nodiscard]] inline std::uint64_t available_physical_bytes() noexcept {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return ::GlobalMemoryStatusEx(&status) != 0 ? status.ullAvailPhys : 0;
#else
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) { return 0; }
    char line[256];
    unsigned long long kib = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) { break; }
    }
    std::fclose(file);
    return static_cast<std::uint64_t>(kib) * 1024ULL;
#endif
}

} // namespace ninfer::os
