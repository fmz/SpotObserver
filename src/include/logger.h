//
// Created by faisa on 5/12/2025.
//

#pragma once
#include <format>
#include <iostream>

namespace SOb {

// Define a function pointer type for logging
typedef void (*LogCallback)(const char* message);

// Store the callback function
extern LogCallback unityLogCallback;
extern bool logging_enabled;

// Function to log messages
template <typename... Args>
void LogMessage(const std::format_string<Args...> fmt, Args&&... args) {
    if (logging_enabled) {
        auto msg = std::format(fmt, std::forward<Args>(args)...);
        std::cout << msg << "\n";
        if (unityLogCallback) {
            unityLogCallback(msg.c_str());
        }
    }
}

// Forward std::string to the template
inline void LogMessage(const std::string& s) {
    LogMessage("{}", s);
}

static void ToggleLogging(bool enable) {
    if (enable) {
        logging_enabled = true;
        LogMessage("Logging enabled.");
    } else {
        LogMessage("Logging disabled.");
        logging_enabled = false;
    }
}

}