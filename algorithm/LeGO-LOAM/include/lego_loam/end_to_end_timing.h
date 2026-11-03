#ifndef END_TO_END_TIMING_H
#define END_TO_END_TIMING_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>

namespace lego_loam_timing {

using Clock = std::chrono::steady_clock;

inline std::atomic<bool>& getEnabledFlag() {
    static std::atomic<bool> enabled(false);
    return enabled;
}

inline void setEnabled(bool enabled) {
    getEnabledFlag().store(enabled);
}

inline bool isEnabled() {
    return getEnabledFlag().load();
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

    // Limita la memoria occupata da frame non completati.
    while (starts.size() > 4096) {
        starts.erase(starts.begin());
    }
}

inline bool take(int32_t sec, uint32_t nanosec,
                 Clock::time_point& start) {
    std::lock_guard<std::mutex> lock(getMutex());

    auto& starts = getStarts();
    const auto key = stampKey(sec, nanosec);
    const auto it = starts.find(key);

    if (it == starts.end()) {
        return false;
    }

    start = it->second;
    starts.erase(it);
    return true;
}

}  // namespace lego_loam_timing

#endif
