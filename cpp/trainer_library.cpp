#include "asm_trainer.h"
#include "trainer_bridge.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

struct AsmTrainerHandle {
    TrainerBridge bridge;
    std::thread worker;
    std::mutex api_mutex;
    AsmTrainerConfig config{};
    std::string spv_dir;
};

namespace {
std::mutex g_singleton_mutex;
AsmTrainerHandle* g_active = nullptr;

void set_error(char* buffer, std::size_t size, const std::string& message) {
    if (buffer == nullptr || size == 0) return;
    std::snprintf(buffer, size, "%s", message.c_str());
}

void normalize_config(AsmTrainerConfig& config) {
    if (config.struct_size == 0) config.struct_size = static_cast<std::uint32_t>(sizeof(AsmTrainerConfig));
    if (config.struct_size < sizeof(AsmTrainerConfig))
        throw std::runtime_error("AsmTrainerConfig 结构体版本过旧");
    if (config.model != 0) throw std::runtime_error("当前共享库只支持 model=0（3B）");
    if (config.optimizer != 0) throw std::runtime_error("当前共享库只支持 optimizer=0（apollo_mini）");
    if (config.param_mode > 1 || config.grad_mode > 1 || config.bf16_scope > 2)
        throw std::runtime_error("param_mode/grad_mode/bf16_scope 取值无效");
    if (!(config.learning_rate > 0.0f)) config.learning_rate = 1e-4f;
    if (!(config.mini_scale > 0.0f)) config.mini_scale = 128.0f;
}

void shutdown_handle(AsmTrainerHandle* handle) {
    if (handle == nullptr) return;
    handle->bridge.request_command(TrainerBridge::Command::Stop);
    if (handle->worker.joinable()) handle->worker.join();
}

void clear_active(AsmTrainerHandle* handle) {
    std::lock_guard<std::mutex> lock(g_singleton_mutex);
    if (g_active == handle) g_active = nullptr;
}
}  // namespace

extern "C" {

ASM_TRAINER_API AsmTrainerHandle* asm_trainer_create(const char* spv_dir,
                                                     const AsmTrainerConfig* config,
                                                     char* error_buffer,
                                                     size_t error_buffer_size) {
    try {
        if (spv_dir == nullptr || *spv_dir == '\0' || config == nullptr)
            throw std::runtime_error("spv_dir/config 不能为空");
        AsmTrainerConfig normalized = *config;
        normalize_config(normalized);

        std::unique_lock<std::mutex> singleton_lock(g_singleton_mutex);
        if (g_active != nullptr) throw std::runtime_error("同一进程只能创建一个 Vulkan trainer");
        std::unique_ptr<AsmTrainerHandle> handle(new AsmTrainerHandle());
        handle->config = normalized;
        handle->spv_dir = spv_dir;
        handle->worker = std::thread([handle = handle.get()] {
            try {
                const int rc = trainer_worker_main(handle->config, handle->spv_dir, handle->bridge);
                if (rc != 0 && !handle->bridge.failed())
                    handle->bridge.fail("trainer worker 退出，rc=" + std::to_string(rc));
            } catch (const std::exception& e) {
                handle->bridge.fail(e.what());
            } catch (...) {
                handle->bridge.fail("trainer worker 发生未知异常");
            }
        });
        g_active = handle.get();
        singleton_lock.unlock();

        if (!handle->bridge.wait_ready()) {
            const std::string error = handle->bridge.error();
            shutdown_handle(handle.get());
            clear_active(handle.get());
            throw std::runtime_error(error.empty() ? "trainer 初始化失败" : error);
        }
        return handle.release();
    } catch (const std::exception& e) {
        set_error(error_buffer, error_buffer_size, e.what());
        return nullptr;
    } catch (...) {
        set_error(error_buffer, error_buffer_size, "trainer 创建发生未知异常");
        return nullptr;
    }
}

ASM_TRAINER_API int asm_trainer_train_step(AsmTrainerHandle* handle,
                                           AsmTrainerStats* stats,
                                           char* error_buffer,
                                           size_t error_buffer_size) {
    if (handle == nullptr || stats == nullptr) {
        set_error(error_buffer, error_buffer_size, "handle/stats 不能为空");
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->api_mutex);
    if (handle->bridge.failed()) {
        set_error(error_buffer, error_buffer_size, handle->bridge.error());
        return -1;
    }
    handle->bridge.request_command(TrainerBridge::Command::Step);
    handle->bridge.wait_result();
    if (handle->bridge.failed()) {
        set_error(error_buffer, error_buffer_size, handle->bridge.error());
        return -1;
    }
    *stats = handle->bridge.last_stats();
    return 0;
}

ASM_TRAINER_API int asm_trainer_get_stats(AsmTrainerHandle* handle,
                                          AsmTrainerStats* stats,
                                          char* error_buffer,
                                          size_t error_buffer_size) {
    if (handle == nullptr || stats == nullptr) {
        set_error(error_buffer, error_buffer_size, "handle/stats 不能为空");
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->api_mutex);
    if (handle->bridge.failed()) {
        set_error(error_buffer, error_buffer_size, handle->bridge.error());
        return -1;
    }
    *stats = handle->bridge.last_stats();
    return 0;
}

ASM_TRAINER_API int asm_trainer_save_checkpoint(AsmTrainerHandle* handle,
                                                const char* path,
                                                char* error_buffer,
                                                size_t error_buffer_size) {
    if (handle == nullptr || path == nullptr || *path == '\0') {
        set_error(error_buffer, error_buffer_size, "handle/path 不能为空");
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->api_mutex);
    if (handle->bridge.failed()) {
        set_error(error_buffer, error_buffer_size, handle->bridge.error());
        return -1;
    }
    handle->bridge.request_save(path);
    handle->bridge.wait_result();
    if (handle->bridge.failed()) {
        set_error(error_buffer, error_buffer_size, handle->bridge.error());
        return -1;
    }
    return 0;
}

ASM_TRAINER_API void asm_trainer_destroy(AsmTrainerHandle* handle) {
    if (handle == nullptr) return;
    shutdown_handle(handle);
    clear_active(handle);
    delete handle;
}

ASM_TRAINER_API const char* asm_trainer_version(void) {
    return ASM_TRAINER_VERSION;
}

}  // extern "C"
