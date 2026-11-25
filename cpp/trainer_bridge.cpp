#include "trainer_bridge.hpp"

#include <utility>

void TrainerBridge::mark_ready() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_ = true;
    }
    cv_.notify_all();
}

void TrainerBridge::fail(const std::string& error) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_.empty()) error_ = error;
        failed_ = true;
        ready_ = true;
        done_ = true;
        command_ = Command::None;
    }
    cv_.notify_all();
}

bool TrainerBridge::wait_ready() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return ready_ || failed_; });
    return !failed_;
}

TrainerBridge::Command TrainerBridge::wait_command() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return command_ != Command::None || failed_; });
    if (failed_) return Command::Stop;
    const Command command = command_;
    command_ = Command::None;
    return command;
}

void TrainerBridge::request_command(Command command) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) return;
        command_ = command;
        if (command == Command::Step) done_ = false;
    }
    cv_.notify_all();
}

void TrainerBridge::publish(const AsmTrainerStats& stats) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = stats;
        done_ = true;
    }
    cv_.notify_all();
}

void TrainerBridge::wait_result() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return done_ || failed_; });
    done_ = false;
}

AsmTrainerStats TrainerBridge::last_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

std::string TrainerBridge::error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

bool TrainerBridge::failed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_;
}
