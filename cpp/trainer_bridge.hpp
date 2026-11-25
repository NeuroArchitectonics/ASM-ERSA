#pragma once

#include "asm_trainer.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

class TrainerBridge {
public:
    enum class Command { None, Step, Save, Load, Stop };

    void mark_ready();
    void fail(const std::string& error);
    bool wait_ready();
    Command wait_command();
    void request_command(Command command);
    void request_save(const std::string& path);
    std::string take_save_path();
    void request_load(const std::string& path);
    std::string take_load_path();
    void publish(const AsmTrainerStats& stats);
    void wait_result();
    AsmTrainerStats last_stats() const;
    std::string error() const;
    bool failed() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    Command command_ = Command::None;
    AsmTrainerStats stats_{};
    std::string error_;
    std::string save_path_;
    std::string load_path_;
    bool ready_ = false;
    bool done_ = false;
    bool failed_ = false;
};

int trainer_worker_main(const AsmTrainerConfig& config,
                        const std::string& spv_dir,
                        TrainerBridge& bridge);
