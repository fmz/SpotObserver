//
// Created by faisa on 5/12/2025.
//

#pragma once
#include <cstdint>
#include <format>
#include <iostream>

namespace SOb {

// Define a function pointer type for logging
typedef void (*LogCallback)(const char* message);

// Logging verbosity. Levels are cumulative: each level includes everything below it.
//   NONE - no logging at all.
//   PERF - only performance logging (timing + memory, via LogPerf).
//   ALL  - everything (LogPerf + general LogMessage).
enum class LogLevel : int32_t {
    NONE = 0,
    PERF = 1,
    ALL  = 2,
};

// Store the callback function
extern LogCallback unityLogCallback;
extern LogLevel    logging_level;

// Emit one already-formatted message to stdout and the Unity callback (if set).
inline void __emitLog(const std::string& msg) {
    std::cout << msg << "\n";
    if (unityLogCallback) {
        unityLogCallback(msg.c_str());
    }
}

// General-purpose logging. Printed only at LogLevel::ALL.
template <typename... Args>
void LogMessage(const std::format_string<Args...> fmt, Args&&... args) {
    if (logging_level >= LogLevel::ALL) {
        __emitLog(std::format(fmt, std::forward<Args>(args)...));
    }
}

// Forward std::string to the template
inline void LogMessage(const std::string& s) {
    LogMessage("{}", s);
}

// Performance logging (timing + memory). Printed at LogLevel::PERF and above, so it can be
// enabled without the noise of general logging.
template <typename... Args>
void LogPerf(const std::format_string<Args...> fmt, Args&&... args) {
    if (logging_level >= LogLevel::PERF) {
        __emitLog(std::format(fmt, std::forward<Args>(args)...));
    }
}

// Forward std::string to the template
inline void LogPerf(const std::string& s) {
    LogPerf("{}", s);
}

static void SetLogLevel(LogLevel level) {
    // Announce the transition through whichever channel can still be heard.
    logging_level = LogLevel::ALL;  // temporarily, so the message below is emitted
    switch (level) {
        case LogLevel::NONE: LogMessage("Logging disabled."); break;
        case LogLevel::PERF: LogMessage("Logging set to PERF (timing + memory only)."); break;
        case LogLevel::ALL:  LogMessage("Logging set to ALL."); break;
    }
    logging_level = level;
}

// Backward-compatible boolean toggle: true -> ALL, false -> NONE.
static void ToggleLogging(bool enable) {
    SetLogLevel(enable ? LogLevel::ALL : LogLevel::NONE);
}

}