#pragma once

// Force-included into vendored sources that call POSIX-only functions, so the vendored trees stay
// untouched on every platform.

#if defined(_WIN32)
#    include <time.h>
// localtime_r(const time_t*, tm*) -> tm* or nullptr, expressed through localtime_s(tm*, const
// time_t*).
#    define localtime_r(timep, result) (::localtime_s((result), (timep)) == 0 ? (result) : nullptr)
#endif
