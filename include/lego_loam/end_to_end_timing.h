#ifndef LEGO_LOAM_END_TO_END_TIMING_H
#define LEGO_LOAM_END_TO_END_TIMING_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>

namespace lego_loam_timing {

using Clock = std::chrono::steady_clock;

// Disabled by default.
inline std::atomic<bool>& enabled() {
    static std::atomic<bool> value{false};
    return value;
}

inline void setEnabled(bool value) {
    enabled().store(value);
}

inline bool isEnabled() {
    return enabled().load();
}

inline std::mutex& getMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::map<int64_t, Clock::time_point>& getStarts() {
    static std::map<int64_t, Clock::time_point> starts;
    return starts;
}

inline int64_t stampKey(int32_t sec, uint32_t nanosec) {
    return static_cast<int64_t>(sec) * 1000000000LL + nanosec;
}

inline void record(int32_t sec, uint32_t nanosec,
                   Clock::time_point start) {
    std::lock_guard<std::mutex> lock(getMutex());

    auto& starts = getStarts();
    starts[stampKey(sec, nanosec)] = start;

    while (starts.size() > 4096) {
        starts.erase(starts.begin());
    }
}

inline bool take(int32_t sec, uint32_t nanosec,
                 Clock::time_point& start) {
    std::lock_guard<std::mutex> lock(getMutex());

    auto& starts = getStarts();
    const auto it = starts.find(stampKey(sec, nanosec));

    if (it == starts.end()) {
        return false;
    }

    start = it->second;
    starts.erase(it);
    return true;
}

}  // namespace lego_loam_timing

#endif
