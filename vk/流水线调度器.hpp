// ============================================================================
//  流水线调度器.hpp
//  Vulkan 底层封装 · 全局流水线调度器（命令流封装 + 二进制 .spv 加载版）
//
//  设计要点：
//    1. 阶段与 Vulkan 队列一一对应；每阶段独立优先队列与工作线程。
//    2. 每个工作线程持有一个 vk底层::命令流，任务录制/提交/等待
//       全部交给 命令流->执行()，调度器不直接调用任何 Vulkan 资源接口。
//    3. 依赖通过任务编号建立；前置全部完成时自动入队。
//    4. 失败/停止时对"等待中"的后继做级联取消。
//    5. 单例，析构内 停止() 兜底。
//    6. 逻辑设备归本类持有：vk底层 负责注册（挑物理设备、开队列族、
//       vkCreateDevice 全在它那边），注册结果交给本类保存并在 停止() 释放。
//       调用方与 vk张量 都只借引用（获取设备()），VkQueue 不再出现在接口上。
//    7. 职责边界：本类只吃 着色编译器 产出的 .spv 二进制（外加同名 .spv.hpp
//       登记的元信息），绝不读 GLSL 源码；设备、命令流（含录制与派发）也归本类；
//       缓冲区之间的拷贝归 vk张量，本类不管数据搬运。
//       录制时按登记的布局一次性绑好描述符集、推送常量与派发规模，
//       vk张量 只负责把自己的缓冲/描述符集交给这里。
//    8. 与 vk底层 的对接只有四处，名字或参数不同只改这四处：
//       注册设备() / 解析队列族() / 取队列() / 内建管线()。
//    9. 张量以 shared_ptr 句柄进入任务闭包，张量析构不会让在途任务踩空。
// ============================================================================
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include <deque>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <algorithm>
#include <stdexcept>

// ===========================================================================
//  vk底层
//
//  原先是单独一个 vk底层.hpp 由使用方提供；现在直接内联在这里，职责只有两件：
//    a) 注册逻辑设备 —— 建 instance、挑物理设备、开队列族、vkCreateDevice；
//    b) 把设备上的四类资源包成 RAII 对象 —— 缓冲、描述符集、管线、命令流。
//
//  生命周期：VkInstance / VkPhysicalDevice / VkDevice / 描述符池 / 集布局 全部
//  归 设备核心 一份持有，四类资源各拿一个 shared_ptr<设备核心>。于是上层无论
//  以什么顺序析构（调度器 停止() 先放设备、张量还活着也算），Vulkan 句柄一定
//  在 vkDestroyDevice 之前先释放，不会出现"设备已毁、资源还在销毁"的踩空。
//
//  描述符约定（与 流水线调度器 的录制路径严格对应）：
//    第 i 个张量的描述符集绑到 set i，每个集里只有一个 storage buffer，
//    且都落在 binding 0。所以 GLSL 写 layout(set=i, binding=0)，
//    管线布局是 N 份完全相同的集布局。分配描述符集时不需要知道下标，
//    这也是 设备::分配缓冲描述符集(缓冲&) 只吃一个缓冲的原因。
// ===========================================================================
namespace vk底层 {

// ---------------------------------------------------------------------------
// 结果码 → 文本
// ---------------------------------------------------------------------------
inline const char* 结果文本(VkResult 结果) {
    switch (结果) {
    case VK_SUCCESS:                              return "VK_SUCCESS";
    case VK_NOT_READY:                            return "VK_NOT_READY";
    case VK_TIMEOUT:                              return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY:             return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:           return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:          return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:                    return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:              return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:              return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:          return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:            return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:            return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:               return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:           return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:                return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY:             return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:        return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION:                  return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_UNKNOWN:                        return "VK_ERROR_UNKNOWN";
    default:                                      return "VK_RESULT(未知码)";
    }
}

inline void 检查结果(VkResult 结果, const char* 动作) {
    if (结果 != VK_SUCCESS) {
        throw std::runtime_error(std::string("vk底层：") + 动作 + " 失败 —— " + 结果文本(结果));
    }
}

// ---------------------------------------------------------------------------
// 校验层：只在调试信使可用时开，VKTS_VALIDATION=0 可关掉
// ---------------------------------------------------------------------------
inline VKAPI_ATTR VkBool32 VKAPI_CALL 调试回调(VkDebugUtilsMessageSeverityFlagBitsEXT 严重度,
                                              VkDebugUtilsMessageTypeFlagsEXT 类型,
                                              const VkDebugUtilsMessengerCallbackDataEXT* 数据,
                                              void*) {
    if (严重度 < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) return VK_FALSE;
    if ((类型 & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) == 0 &&
        严重度 < VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) return VK_FALSE;
    std::fprintf(stderr, "[vk底层校验] %s\n",
                 (数据 && 数据->pMessage) ? 数据->pMessage : "(无消息)");
    return VK_FALSE;
}

inline bool 校验层可用() {
    const char* 开关 = std::getenv("VKTS_VALIDATION");
    if (开关 != nullptr && 开关[0] == '0') return false;

    std::uint32_t 数量 = 0;
    vkEnumerateInstanceLayerProperties(&数量, nullptr);
    if (数量 == 0) return false;
    std::vector<VkLayerProperties> 层(数量);
    vkEnumerateInstanceLayerProperties(&数量, 层.data());
    for (const auto& 项 : 层) {
        if (std::strcmp(项.layerName, "VK_LAYER_KHRONOS_validation") == 0) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// 设备核心：instance / 物理设备 / 逻辑设备 / 描述符池 / 集布局 的唯一所有者
// ---------------------------------------------------------------------------
struct 设备核心 {
    static constexpr std::uint32_t 每池集数 = 4096u;

    VkInstance       实例     = VK_NULL_HANDLE;
    VkPhysicalDevice 物理设备 = VK_NULL_HANDLE;
    VkDevice         设备     = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties       设备属性{};
    VkPhysicalDeviceMemoryProperties 内存属性{};
    std::vector<VkQueueFamilyProperties> 队列族属性;
    std::vector<std::uint32_t>           每族队列数;   // 建逻辑设备时实际开了几条

    VkDebugUtilsMessengerEXT              调试信使   = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT   销毁信使   = nullptr;

    std::vector<VkDescriptorPool> 描述符池列表;
    VkDescriptorPool              当前池     = VK_NULL_HANDLE;
    std::uint32_t                 当前池余量 = 0;
    VkDescriptorSetLayout         集布局     = VK_NULL_HANDLE;   // 单 binding 0，所有 set 共用
    std::recursive_mutex          描述符锁;

    ~设备核心() {
        if (设备 != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(设备);
            for (VkDescriptorPool 池 : 描述符池列表) vkDestroyDescriptorPool(设备, 池, nullptr);
            if (集布局 != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(设备, 集布局, nullptr);
            vkDestroyDevice(设备, nullptr);
        }
        if (调试信使 != VK_NULL_HANDLE && 销毁信使 != nullptr) {
            销毁信使(实例, 调试信使, nullptr);
        }
        if (实例 != VK_NULL_HANDLE) vkDestroyInstance(实例, nullptr);
    }

    // 要求位必须全有，期望位有则加分：统一内存的 iGPU 上 DEVICE_LOCAL|HOST_VISIBLE
    // 是最优解（一次映射，GPU 也当本地内存用），拿不到再退纯 HOST_VISIBLE。
    // 需要字节 > 0 时跳过装不下的堆 —— 否则会选中小的 DEVICE_LOCAL 可见堆（如 256MB
    // carveout），大权重第一次分配就超限。
    std::uint32_t 找内存类型(std::uint32_t 类型位,
                            VkMemoryPropertyFlags 要求,
                            VkMemoryPropertyFlags 期望,
                            VkDeviceSize 需要字节 = 0) const {
        std::uint32_t 最大备选 = UINT32_MAX;
        VkDeviceSize  最大备选堆 = 0;
        for (std::uint32_t i = 0; i < 内存属性.memoryTypeCount; ++i) {
            if ((类型位 & (1u << i)) == 0) continue;
            const VkMemoryPropertyFlags 旗标 = 内存属性.memoryTypes[i].propertyFlags;
            if ((旗标 & 要求) != 要求) continue;
            const VkDeviceSize 堆大小 =
                内存属性.memoryHeaps[内存属性.memoryTypes[i].heapIndex].size;
            if (需要字节 != 0 && 堆大小 < 需要字节) continue;
            if ((旗标 & 期望) == 期望) return i;
            if (堆大小 > 最大备选堆) { 最大备选堆 = 堆大小; 最大备选 = i; }
        }
        return 最大备选;
    }

    // 主机可见内存的候选顺序：大缓冲优先最大的可见堆（APU 的 256MB
    // DEVICE_LOCAL 可见 carveout 装不下整模型权重），小缓冲优先 DEVICE_LOCAL。
    // 分配失败时按顺序回退，避免把 5GB 级权重塞进小堆。
    std::vector<std::uint32_t> 候选内存类型(std::uint32_t 类型位,
                                           VkDeviceSize 需要字节,
                                           bool 读多 = false) const {
        const VkMemoryPropertyFlags 主机可见 = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        const VkMemoryPropertyFlags 主机一致 = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const bool 大缓冲 = 需要字节 >= (VkDeviceSize)64 << 20;
        struct 候选 { std::uint32_t 类型; VkDeviceSize 堆; bool 本地; bool 缓存; };
        std::vector<候选> 列表;
        bool 有缓存 = false;
        for (std::uint32_t i = 0; i < 内存属性.memoryTypeCount; ++i) {
            if ((类型位 & (1u << i)) == 0) continue;
            const VkMemoryPropertyFlags 旗标 = 内存属性.memoryTypes[i].propertyFlags;
            if ((旗标 & 主机可见) != 主机可见) continue;
            if (!读多 && (旗标 & 主机一致) != 主机一致) continue;
            const VkDeviceSize 堆大小 =
                内存属性.memoryHeaps[内存属性.memoryTypes[i].heapIndex].size;
            if (需要字节 != 0 && 堆大小 < 需要字节) continue;
            const bool 缓存 = (旗标 & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
            列表.push_back({i, 堆大小,
                            (旗标 & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
                            缓存});
            if (缓存) 有缓存 = true;
        }
        std::sort(列表.begin(), 列表.end(),
                  [大缓冲, 读多, 有缓存](const 候选& a, const 候选& b) {
            if (读多 && 有缓存) {
                if (a.本地 != b.本地) return a.本地;
                if (a.缓存 != b.缓存) return a.缓存;
                return a.堆 > b.堆;
            }
            if (大缓冲) {
                if (a.堆 != b.堆) return a.堆 > b.堆;
                return a.本地 && !b.本地;
            }
            if (a.本地 != b.本地) return a.本地;
            return a.堆 > b.堆;
        });
        std::vector<std::uint32_t> 结果;
        结果.reserve(列表.size());
        for (const 候选& c : 列表) 结果.push_back(c.类型);
        return 结果;
    }

    VkDescriptorPool 开新池() {
        VkDescriptorPoolSize 规格{};
        规格.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        规格.descriptorCount = 每池集数;

        VkDescriptorPoolCreateInfo 信息{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        信息.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        信息.maxSets       = 每池集数;
        信息.poolSizeCount = 1;
        信息.pPoolSizes    = &规格;

        VkDescriptorPool 池 = VK_NULL_HANDLE;
        检查结果(vkCreateDescriptorPool(设备, &信息, nullptr, &池), "创建描述符池");
        描述符池列表.push_back(池);
        return 池;
    }

    VkDescriptorSetLayout 取集布局() {
        std::lock_guard<std::recursive_mutex> 守卫(描述符锁);
        if (集布局 != VK_NULL_HANDLE) return 集布局;

        VkDescriptorSetLayoutBinding 绑定{};
        绑定.binding            = 0;
        绑定.descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        绑定.descriptorCount    = 1;
        绑定.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo 信息{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        信息.bindingCount = 1;
        信息.pBindings    = &绑定;
        检查结果(vkCreateDescriptorSetLayout(设备, &信息, nullptr, &集布局), "创建描述符集布局");
        return 集布局;
    }
};

// ---------------------------------------------------------------------------
// 缓冲：存储缓冲 + 常驻映射的主机可见内存
// ---------------------------------------------------------------------------
class 缓冲 {
public:
    缓冲(std::shared_ptr<设备核心> 核心, std::size_t 字节, bool 读多 = false)
        : 核心_(std::move(核心)), 字节_(字节)
    {
        if (字节_ == 0) throw std::runtime_error("vk底层：缓冲字节数为 0");

        VkBufferCreateInfo 信息{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        信息.size        = static_cast<VkDeviceSize>(字节_);
        信息.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        信息.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        检查结果(vkCreateBuffer(核心_->设备, &信息, nullptr, &缓冲_), "创建存储缓冲");

        VkMemoryRequirements 需求{};
        vkGetBufferMemoryRequirements(核心_->设备, 缓冲_, &需求);

        const auto 候选 = 核心_->候选内存类型(需求.memoryTypeBits, 需求.size, 读多);
        if (候选.empty()) {
            vkDestroyBuffer(核心_->设备, 缓冲_, nullptr);
            缓冲_ = VK_NULL_HANDLE;
            throw std::runtime_error("vk底层：找不到主机可见的内存类型");
        }

        VkMemoryAllocateInfo 分配{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        分配.allocationSize = 需求.size;
        VkResult 最后结果 = VK_SUCCESS;
        std::uint32_t 选中类型 = 候选.front();
        for (std::uint32_t 类型 : 候选) {
            分配.memoryTypeIndex = 类型;
            最后结果 = vkAllocateMemory(核心_->设备, &分配, nullptr, &内存_);
            if (最后结果 == VK_SUCCESS) { 选中类型 = 类型; break; }
            if (最后结果 != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
                最后结果 != VK_ERROR_OUT_OF_HOST_MEMORY) break;
        }
        if (最后结果 != VK_SUCCESS) {
            std::fprintf(stderr, "[vk底层] 分配失败 %.1fMB 类型hints=0x%x 存活缓冲≈%.1fGB：%s\n",
                         (double)需求.size / (1024.0 * 1024.0), 需求.memoryTypeBits,
                         (double)存活字节计().load() / (1024.0 * 1024.0 * 1024.0),
                         结果文本(最后结果));
        }
        try {
            检查结果(最后结果, "分配缓冲内存");
            检查结果(vkBindBufferMemory(核心_->设备, 缓冲_, 内存_, 0), "绑定缓冲内存");
            检查结果(vkMapMemory(核心_->设备, 内存_, 0, 需求.size, 0, &映射_), "映射缓冲内存");
            分配字节_ = (std::size_t)需求.size;
            内存旗标_ = 核心_->内存属性.memoryTypes[选中类型].propertyFlags;
            存活字节计().fetch_add((long long)需求.size);
            static std::atomic<int> 读多调试{0};
            if (读多 && 需求.size >= (VkDeviceSize)1 << 20 &&
                读多调试.fetch_add(1) < 4) {
                const auto& 类型信息 = 核心_->内存属性.memoryTypes[选中类型];
                const double 堆MB = double(核心_->内存属性.memoryHeaps[类型信息.heapIndex].size) /
                                    (1024.0 * 1024.0);
                std::fprintf(stderr,
                             "[mem] read-heavy %.1f MB type=%u flags=0x%x heap=%.1f MB\n",
                             double(需求.size) / (1024.0 * 1024.0), 选中类型,
                             (unsigned)内存旗标_, 堆MB);
            }
        } catch (...) {
            if (内存_ != VK_NULL_HANDLE) vkFreeMemory(核心_->设备, 内存_, nullptr);
            vkDestroyBuffer(核心_->设备, 缓冲_, nullptr);
            throw;
        }
    }

    ~缓冲() {
        if (内存_ != VK_NULL_HANDLE) 存活字节计().fetch_sub((long long)分配字节_);
        if (映射_ != nullptr) vkUnmapMemory(核心_->设备, 内存_);
        if (内存_ != VK_NULL_HANDLE) vkFreeMemory(核心_->设备, 内存_, nullptr);
        if (缓冲_ != VK_NULL_HANDLE) vkDestroyBuffer(核心_->设备, 缓冲_, nullptr);
    }

    static std::atomic<long long>& 存活字节计() { static std::atomic<long long> v{0}; return v; }

    缓冲(const 缓冲&) = delete;
    缓冲& operator=(const 缓冲&) = delete;

    VkBuffer    句柄()   const { return 缓冲_; }
    std::size_t 字节数() const { return 字节_; }
    void*       映射()   const { return 映射_; }

    void 写入(const void* 数据, std::size_t 字节) {
        if (字节 > 字节_) throw std::runtime_error("vk底层：写入超出缓冲容量");
        std::memcpy(映射_, 数据, 字节);
        if ((内存旗标_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange 范围{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            范围.memory = 内存_;
            范围.offset = 0;
            范围.size   = VK_WHOLE_SIZE;
            vkFlushMappedMemoryRanges(核心_->设备, 1, &范围);
        }
    }

    void 读取(void* 数据, std::size_t 字节) const {
        if (字节 > 字节_) throw std::runtime_error("vk底层：读取超出缓冲容量");
        if ((内存旗标_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange 范围{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            范围.memory = 内存_;
            范围.offset = 0;
            范围.size   = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(核心_->设备, 1, &范围);
        }
        std::memcpy(数据, 映射_, 字节);
    }

private:
    std::shared_ptr<设备核心> 核心_;
    VkBuffer                  缓冲_ = VK_NULL_HANDLE;
    VkDeviceMemory            内存_ = VK_NULL_HANDLE;
    void*                     映射_ = nullptr;
    std::size_t               字节_ = 0;
    std::size_t               分配字节_ = 0;
    VkMemoryPropertyFlags     内存旗标_ = 0;
};

// ---------------------------------------------------------------------------
// 描述符集：一个集一个 storage buffer，落在 binding 0
// ---------------------------------------------------------------------------
class 描述符集 {
public:
    static std::shared_ptr<描述符集> 创建(const std::shared_ptr<设备核心>& 核心, 缓冲& 目标) {
        std::lock_guard<std::recursive_mutex> 守卫(核心->描述符锁);
        if (核心->当前池 == VK_NULL_HANDLE || 核心->当前池余量 == 0) {
            核心->当前池     = 核心->开新池();
            核心->当前池余量 = 设备核心::每池集数;
        }

        VkDescriptorSetLayout 布局 = 核心->集布局 != VK_NULL_HANDLE ? 核心->集布局
                                                                    : 核心->取集布局();
        VkDescriptorSetAllocateInfo 信息{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        信息.descriptorPool     = 核心->当前池;
        信息.descriptorSetCount = 1;
        信息.pSetLayouts        = &布局;

        VkDescriptorSet 集 = VK_NULL_HANDLE;
        VkResult 结果 = vkAllocateDescriptorSets(核心->设备, &信息, &集);
        if (结果 == VK_ERROR_OUT_OF_POOL_MEMORY || 结果 == VK_ERROR_FRAGMENTED_POOL) {
            核心->当前池     = 核心->开新池();      // 池用尽：再开一个，本次重分配
            核心->当前池余量 = 设备核心::每池集数;
            信息.descriptorPool = 核心->当前池;
            结果 = vkAllocateDescriptorSets(核心->设备, &信息, &集);
        }
        if (结果 != VK_SUCCESS) throw std::runtime_error(std::string("vk底层：分配描述符集失败 —— ")
                                                        + 结果文本(结果));
        --核心->当前池余量;

        VkDescriptorBufferInfo 缓冲信息{};
        缓冲信息.buffer = 目标.句柄();
        缓冲信息.offset = 0;
        缓冲信息.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet 写入{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        写入.dstSet          = 集;
        写入.dstBinding      = 0;
        写入.descriptorCount = 1;
        写入.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        写入.pBufferInfo     = &缓冲信息;
        vkUpdateDescriptorSets(核心->设备, 1, &写入, 0, nullptr);

        return std::shared_ptr<描述符集>(new 描述符集(核心, 核心->当前池, 集));
    }

    ~描述符集() {
        // 池由 设备核心 持有，核心一定比本对象活得久（shared_ptr 计数保证）。
        // 释放与分配共用同一把锁：预载线程与 TE 主线程会并发建/销毁张量。
        std::lock_guard<std::recursive_mutex> 守卫(核心_->描述符锁);
        vkFreeDescriptorSets(核心_->设备, 池_, 1, &集_);
    }

    描述符集(const 描述符集&) = delete;
    描述符集& operator=(const 描述符集&) = delete;

    VkDescriptorSet 句柄() const { return 集_; }

private:
    描述符集(std::shared_ptr<设备核心> 核心, VkDescriptorPool 池, VkDescriptorSet 集)
        : 核心_(std::move(核心)), 池_(池), 集_(集) {}

    std::shared_ptr<设备核心> 核心_;
    VkDescriptorPool          池_ = VK_NULL_HANDLE;
    VkDescriptorSet           集_ = VK_NULL_HANDLE;
};

// ---------------------------------------------------------------------------
// 管线：一份 SPIR-V + N 个相同集布局（set i 的 binding 0）+ 一段推送常量
// ---------------------------------------------------------------------------
class 管线 {
public:
    管线(std::shared_ptr<设备核心> 核心,
         const std::vector<std::uint32_t>& 字,
         std::uint32_t 绑定数,
         std::uint32_t 工作组大小,
         std::uint32_t 推送常量字节)
        : 核心_(std::move(核心)), 绑定数_(绑定数), 工作组大小_(工作组大小)
    {
        if (字.empty()) throw std::runtime_error("vk底层：SPIR-V 字流为空");
        if (绑定数_ == 0) throw std::runtime_error("vk底层：绑定数为 0，没有可绑的张量");

        VkShaderModuleCreateInfo 模块信息{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        模块信息.codeSize = 字.size() * sizeof(std::uint32_t);
        模块信息.pCode    = 字.data();
        检查结果(vkCreateShaderModule(核心_->设备, &模块信息, nullptr, &模块_), "创建着色器模块");

        VkDescriptorSetLayout 布局 = 核心_->取集布局();
        std::vector<VkDescriptorSetLayout> 集布局(绑定数_, 布局);

        VkPushConstantRange 推送{};
        推送.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        推送.offset     = 0;
        推送.size       = 推送常量字节;

        VkPipelineLayoutCreateInfo 布局信息{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        布局信息.setLayoutCount = 绑定数_;
        布局信息.pSetLayouts    = 集布局.data();
        if (推送常量字节 > 0) {
            if (推送常量字节 > 核心_->设备属性.limits.maxPushConstantsSize) {
                vkDestroyShaderModule(核心_->设备, 模块_, nullptr);
                throw std::runtime_error("vk底层：推送常量 " + std::to_string(推送常量字节)
                                         + " 字节超过设备上限 "
                                         + std::to_string(核心_->设备属性.limits.maxPushConstantsSize));
            }
            布局信息.pushConstantRangeCount = 1;
            布局信息.pPushConstantRanges    = &推送;
        }
        检查结果(vkCreatePipelineLayout(核心_->设备, &布局信息, nullptr, &布局_), "创建管线布局");

        VkComputePipelineCreateInfo 管线信息{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        管线信息.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        管线信息.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        管线信息.stage.module = 模块_;
        管线信息.stage.pName  = "main";
        管线信息.layout       = 布局_;

        VkPipeline 管线句柄 = VK_NULL_HANDLE;
        VkResult 结果 = vkCreateComputePipelines(核心_->设备, VK_NULL_HANDLE, 1, &管线信息,
                                                nullptr, &管线句柄);
        if (结果 != VK_SUCCESS) {
            vkDestroyPipelineLayout(核心_->设备, 布局_, nullptr);
            vkDestroyShaderModule(核心_->设备, 模块_, nullptr);
            throw std::runtime_error(std::string("vk底层：创建计算管线失败 —— ") + 结果文本(结果));
        }
        管线_ = 管线句柄;
    }

    ~管线() {
        if (管线_ != VK_NULL_HANDLE) vkDestroyPipeline(核心_->设备, 管线_, nullptr);
        if (布局_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(核心_->设备, 布局_, nullptr);
        if (模块_ != VK_NULL_HANDLE) vkDestroyShaderModule(核心_->设备, 模块_, nullptr);
    }

    管线(const 管线&) = delete;
    管线& operator=(const 管线&) = delete;

    VkPipeline       管线句柄() const { return 管线_; }
    VkPipelineLayout 布局句柄() const { return 布局_; }
    std::uint32_t    绑定数()   const { return 绑定数_; }
    std::uint32_t    工作组大小() const { return 工作组大小_; }

private:
    std::shared_ptr<设备核心> 核心_;
    VkShaderModule            模块_ = VK_NULL_HANDLE;
    VkPipelineLayout          布局_ = VK_NULL_HANDLE;
    VkPipeline                管线_ = VK_NULL_HANDLE;
    std::uint32_t             绑定数_     = 0;
    std::uint32_t             工作组大小_ = 0;
};

// 多个命令流可能共享同一个 VkQueue（同一阶段的多个 worker）。
// Vulkan 要求 VkQueue 外部同步，因此所有 vkQueueSubmit 共用一把提交锁。
// 锁只覆盖提交调用，命令的 GPU 执行仍可在不同队列上重叠。
inline std::mutex& 全局队列提交锁() {
    static std::mutex 锁;
    return 锁;
}

// ---------------------------------------------------------------------------
// 命令流：一条命令缓冲 + 围栏，执行() 走完 分配→录制→提交→等待→复位
// 一个命令流只归一个线程用（调度器每条工作线程独占一条）
// ---------------------------------------------------------------------------
class 命令流 {
public:
    命令流(std::shared_ptr<设备核心> 核心, std::uint32_t 族索引, VkQueue 队列)
        : 核心_(std::move(核心)), 队列_(队列)
    {
        if (队列_ == VK_NULL_HANDLE) throw std::runtime_error("vk底层：命令流需要有效队列");

        VkCommandPoolCreateInfo 池信息{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        池信息.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        池信息.queueFamilyIndex = 族索引;
        检查结果(vkCreateCommandPool(核心_->设备, &池信息, nullptr, &池_), "创建命令池");

        VkCommandBufferAllocateInfo 分配{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        分配.commandPool        = 池_;
        分配.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        分配.commandBufferCount = 1;
        检查结果(vkAllocateCommandBuffers(核心_->设备, &分配, &命令缓冲_), "分配命令缓冲");

        VkFenceCreateInfo 围栏信息{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        检查结果(vkCreateFence(核心_->设备, &围栏信息, nullptr, &围栏_), "创建围栏");
    }

    ~命令流() {
        if (池_ != VK_NULL_HANDLE) vkDestroyCommandPool(核心_->设备, 池_, nullptr);  // 连带命令缓冲
        if (围栏_ != VK_NULL_HANDLE) vkDestroyFence(核心_->设备, 围栏_, nullptr);
    }

    命令流(const 命令流&) = delete;
    命令流& operator=(const 命令流&) = delete;

    // 录制回调抛异常时也要把命令缓冲收回可用态，否则这条流之后就废了
    void 执行(const std::function<void(VkCommandBuffer)>& 录制) {
        std::lock_guard<std::mutex> 守卫(锁_);
        检查结果(vkResetFences(核心_->设备, 1, &围栏_), "复位围栏");
        VkCommandBufferBeginInfo 开始{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        开始.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        检查结果(vkBeginCommandBuffer(命令缓冲_, &开始), "开始录制命令缓冲");

        try {
            录制(命令缓冲_);
            检查结果(vkEndCommandBuffer(命令缓冲_), "结束录制命令缓冲");
        } catch (...) {
            vkResetCommandBuffer(命令缓冲_, 0);
            throw;
        }

        VkSubmitInfo 提交{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        提交.commandBufferCount = 1;
        提交.pCommandBuffers    = &命令缓冲_;
        {
            std::lock_guard<std::mutex> 队列守卫(全局队列提交锁());
            检查结果(vkQueueSubmit(队列_, 1, &提交, 围栏_), "提交命令缓冲");
        }
        检查结果(vkWaitForFences(核心_->设备, 1, &围栏_, VK_TRUE, UINT64_MAX), "等待围栏");
    }

private:
    std::shared_ptr<设备核心> 核心_;
    VkCommandPool             池_       = VK_NULL_HANDLE;
    VkCommandBuffer           命令缓冲_ = VK_NULL_HANDLE;
    VkFence                   围栏_     = VK_NULL_HANDLE;
    VkQueue                   队列_     = VK_NULL_HANDLE;
    std::mutex                锁_;
};

// ---------------------------------------------------------------------------
// 设备：调度器与 vk张量 唯一打交道的入口
// ---------------------------------------------------------------------------
class 设备 {
public:
    explicit 设备(std::shared_ptr<设备核心> 核心) : 核心_(std::move(核心)) {}

    const std::shared_ptr<设备核心>& 核心() const { return 核心_; }
    VkDevice 句柄() const { return 核心_->设备; }
    const char* 名称() const { return 核心_->设备属性.deviceName; }
    const VkPhysicalDeviceProperties& 属性() const { return 核心_->设备属性; }
    const VkPhysicalDeviceLimits& 上限() const { return 核心_->设备属性.limits; }

    // ② 用途 → 队列族索引：要求位必须齐；纯传输用途优先给专用传输族
    std::uint32_t 获取队列族(VkQueueFlags 用途) const {
        std::uint32_t 命中 = VK_QUEUE_FAMILY_IGNORED;
        for (std::uint32_t i = 0; i < 核心_->队列族属性.size(); ++i) {
            const VkQueueFamilyProperties& 族 = 核心_->队列族属性[i];
            if (族.queueCount == 0) continue;
            if ((族.queueFlags & 用途) != 用途) continue;
            if (用途 == VK_QUEUE_TRANSFER_BIT && 族.queueFlags == VK_QUEUE_TRANSFER_BIT) return i;
            if (命中 == VK_QUEUE_FAMILY_IGNORED) 命中 = i;
        }
        return 命中;
    }

    // ③ 族索引 + 序号 → VkQueue；越界一律给 VK_NULL_HANDLE，由调度器报错
    VkQueue 获取队列(std::uint32_t 族索引, std::uint32_t 序号) const {
        if (族索引 == VK_QUEUE_FAMILY_IGNORED ||
            族索引 >= static_cast<std::uint32_t>(核心_->每族队列数.size())) return VK_NULL_HANDLE;
        if (序号 >= 核心_->每族队列数[族索引]) return VK_NULL_HANDLE;
        VkQueue 队列 = VK_NULL_HANDLE;
        vkGetDeviceQueue(核心_->设备, 族索引, 序号, &队列);
        return 队列;
    }

    std::shared_ptr<命令流> 创建命令流(std::uint32_t 族索引, VkQueue 队列) {
        return std::make_shared<命令流>(核心_, 族索引, 队列);
    }

    // ④ SPIR-V → 计算管线
    std::shared_ptr<管线> 创建计算管线(const std::vector<std::uint32_t>& 字,
                                     std::uint32_t 绑定数,
                                     std::uint32_t 工作组大小,
                                     std::uint32_t 推送常量字节) {
        return std::make_shared<管线>(核心_, 字, 绑定数, 工作组大小, 推送常量字节);
    }

    std::shared_ptr<缓冲> 创建存储缓冲(std::size_t 字节, bool 读多 = false) {
        return std::make_shared<缓冲>(核心_, 字节, 读多);
    }

    std::shared_ptr<描述符集> 分配缓冲描述符集(缓冲& 目标) {
        return 描述符集::创建(核心_, 目标);
    }

private:
    std::shared_ptr<设备核心> 核心_;
};

// ---------------------------------------------------------------------------
// ① 注册逻辑设备：挑物理设备、开队列族、vkCreateDevice 全在这里
// ---------------------------------------------------------------------------
inline std::shared_ptr<设备> 注册逻辑设备(const std::string& 设备名,
                                        std::uint32_t 物理设备索引,
                                        VkQueueFlags 队列用途,
                                        const std::vector<const char*>& 设备扩展,
                                        const std::string& 应用名,
                                        bool 允许软件实现) {
    auto 核心 = std::make_shared<设备核心>();

    // ---- instance ----
    std::vector<const char*> 实例层;
    std::vector<const char*> 实例扩展;
    const bool 开校验 = 校验层可用();
    if (开校验) {
        实例层.push_back("VK_LAYER_KHRONOS_validation");
        实例扩展.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo 应用{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    应用.pApplicationName = 应用名.empty() ? "vk张量系统" : 应用名.c_str();
    应用.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    应用.pEngineName        = "vk张量系统";
    应用.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    应用.apiVersion         = VK_API_VERSION_1_2;

    VkInstanceCreateInfo 实例信息{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    实例信息.pApplicationInfo        = &应用;
    实例信息.enabledLayerCount       = static_cast<std::uint32_t>(实例层.size());
    实例信息.ppEnabledLayerNames     = 实例层.data();
    实例信息.enabledExtensionCount   = static_cast<std::uint32_t>(实例扩展.size());
    实例信息.ppEnabledExtensionNames = 实例扩展.data();
    检查结果(vkCreateInstance(&实例信息, nullptr, &核心->实例), "创建 VkInstance");

    if (开校验) {
        auto 建信使 = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(核心->实例, "vkCreateDebugUtilsMessengerEXT"));
        核心->销毁信使 = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(核心->实例, "vkDestroyDebugUtilsMessengerEXT"));
        if (建信使 != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT 信使信息{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            信使信息.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            信使信息.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            信使信息.pfnUserCallback = 调试回调;
            if (建信使(核心->实例, &信使信息, nullptr, &核心->调试信使) != VK_SUCCESS) {
                核心->调试信使 = VK_NULL_HANDLE;   // 建不成就静默关掉，不影响计算
            }
        }
    }

    // ---- 物理设备 ----
    std::uint32_t 数量 = 0;
    检查结果(vkEnumeratePhysicalDevices(核心->实例, &数量, nullptr), "枚举物理设备");
    if (数量 == 0) throw std::runtime_error("vk底层：这台机器上没有 Vulkan 物理设备");
    std::vector<VkPhysicalDevice> 候选(数量);
    检查结果(vkEnumeratePhysicalDevices(核心->实例, &数量, 候选.data()), "枚举物理设备");

    auto 用途可满足 = [&](VkPhysicalDevice 物理) {
        std::uint32_t 族数 = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(物理, &族数, nullptr);
        if (族数 == 0) return false;
        std::vector<VkQueueFamilyProperties> 族(族数);
        vkGetPhysicalDeviceQueueFamilyProperties(物理, &族数, 族.data());
        for (const auto& 项 : 族) {
            if (项.queueCount > 0 && (项.queueFlags & 队列用途) == 队列用途) return true;
        }
        return false;
    };

    VkPhysicalDevice 选中 = VK_NULL_HANDLE;
    if (!设备名.empty()) {
        for (VkPhysicalDevice 物理 : 候选) {
            VkPhysicalDeviceProperties 属性{};
            vkGetPhysicalDeviceProperties(物理, &属性);
            if (std::strstr(属性.deviceName, 设备名.c_str()) != nullptr) { 选中 = 物理; break; }
        }
        if (选中 == VK_NULL_HANDLE) {
            std::string 清单;
            for (VkPhysicalDevice 物理 : 候选) {
                VkPhysicalDeviceProperties 属性{};
                vkGetPhysicalDeviceProperties(物理, &属性);
                清单 += std::string("\n    ") + 属性.deviceName;
            }
            throw std::runtime_error("vk底层：找不到名字含 \"" + 设备名 + "\" 的设备，现有设备：" + 清单);
        }
    } else {
        std::vector<VkPhysicalDevice> 可用;
        for (VkPhysicalDevice 物理 : 候选) {
            VkPhysicalDeviceProperties 属性{};
            vkGetPhysicalDeviceProperties(物理, &属性);
            if (!允许软件实现 && 属性.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
            if (!用途可满足(物理)) continue;
            可用.push_back(物理);
        }
        if (可用.empty()) {
            throw std::runtime_error("vk底层：没有满足队列用途 0x"
                                     + std::to_string(static_cast<unsigned>(队列用途))
                                     + " 的物理设备（软件实现"
                                     + (允许软件实现 ? "已允许但仍无)" : "未被允许)"));
        }
        if (物理设备索引 >= static_cast<std::uint32_t>(可用.size())) 物理设备索引 = 0;
        选中 = 可用[物理设备索引];
    }
    if (!用途可满足(选中)) {
        VkPhysicalDeviceProperties 属性{};
        vkGetPhysicalDeviceProperties(选中, &属性);
        throw std::runtime_error(std::string("vk底层：设备 ") + 属性.deviceName
                                 + " 没有满足所请求队列用途的队列族");
    }

    核心->物理设备 = 选中;
    vkGetPhysicalDeviceProperties(选中, &核心->设备属性);
    vkGetPhysicalDeviceMemoryProperties(选中, &核心->内存属性);
    std::uint32_t 族数 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(选中, &族数, nullptr);
    核心->队列族属性.resize(族数);
    vkGetPhysicalDeviceQueueFamilyProperties(选中, &族数, 核心->队列族属性.data());

    // ---- 逻辑设备：所有有队列的族都开上，阶段用途到族的解析才不会落空 ----
    std::vector<VkDeviceQueueCreateInfo> 队列创建;
    std::vector<std::vector<float>>      优先级表;
    核心->每族队列数.assign(族数, 0);
    for (std::uint32_t i = 0; i < 族数 && i < 8u; ++i) {
        const std::uint32_t 条数 = 核心->队列族属性[i].queueCount;
        if (条数 == 0) continue;
        const std::uint32_t 实开 = 条数 < 8u ? 条数 : 8u;
        优先级表.emplace_back(实开, 1.0f);
        VkDeviceQueueCreateInfo 项{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        项.queueFamilyIndex = i;
        项.queueCount       = 实开;
        项.pQueuePriorities = 优先级表.back().data();
        队列创建.push_back(项);
        核心->每族队列数[i] = 实开;
    }
    if (队列创建.empty()) throw std::runtime_error("vk底层：物理设备没有任何可用队列族");

    VkPhysicalDeviceFeatures 特性{};   // 计算着色器不需要额外特性，全默认
    特性.shaderInt16 = VK_TRUE;        // bf16 梯度内核用 uint16_t（GL_EXT_shader_explicit_arithmetic_types_int16）

    // Vulkan 1.2 特性：启用 8-bit storage buffer 访问（gather_single/gather_time/scatter_time 需要）
    VkPhysicalDeviceVulkan12Features vulkan12特性{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    vulkan12特性.storageBuffer8BitAccess = VK_TRUE;
    vulkan12特性.shaderInt8 = VK_TRUE;   // INT8 AdamW / 量化算子需要
    vulkan12特性.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features vulkan11特性{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    vulkan11特性.storageBuffer16BitAccess = VK_TRUE;
    vulkan11特性.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    vulkan12特性.pNext = &vulkan11特性;


    VkDeviceCreateInfo 设备信息{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    设备信息.queueCreateInfoCount    = static_cast<std::uint32_t>(队列创建.size());
    设备信息.pQueueCreateInfos        = 队列创建.data();
    设备信息.enabledExtensionCount   = static_cast<std::uint32_t>(设备扩展.size());
    设备信息.ppEnabledExtensionNames = 设备扩展.data();
    设备信息.pEnabledFeatures        = &特性;
    设备信息.pNext                   = &vulkan12特性;  // 链式扩展：Vulkan 1.2 特性
    检查结果(vkCreateDevice(核心->物理设备, &设备信息, nullptr, &核心->设备), "创建逻辑设备");

    return std::make_shared<设备>(核心);
}

} // namespace vk底层

namespace vk张量系统 {

// ---------------------------------------------------------------------------
// 流水线阶段：对应 Vulkan 队列族用途
// ---------------------------------------------------------------------------
enum class 流水线阶段 : std::uint8_t {
    图形 = 0,
    计算 = 1,
    传输 = 2,
    阶段数量 = 3,
};

inline const char* 流水线阶段名(流水线阶段 阶段) {
    switch (阶段) {
    case 流水线阶段::图形: return "图形";
    case 流水线阶段::计算: return "计算";
    case 流水线阶段::传输: return "传输";
    default:               return "未知";
    }
}

// ---------------------------------------------------------------------------
// 任务生命周期状态
// ---------------------------------------------------------------------------
enum class 任务状态 : std::uint8_t {
    等待 = 0,
    就绪 = 1,
    执行中 = 2,
    完成 = 3,
    失败 = 4,
    取消 = 5,
};

inline const char* 任务状态名(任务状态 状态) {
    switch (状态) {
    case 任务状态::等待:   return "等待";
    case 任务状态::就绪:   return "就绪";
    case 任务状态::执行中: return "执行中";
    case 任务状态::完成:   return "完成";
    case 任务状态::失败:   return "失败";
    case 任务状态::取消:   return "取消";
    }
    return "未知";
}

// ---------------------------------------------------------------------------
// 阶段配置：只说"这个阶段开几条并发命令流"，队列由调度器向 vk底层 取
// ---------------------------------------------------------------------------
struct 阶段配置 {
    // 队列族索引留给 vk底层 按阶段用途解析（物理设备属性只有它知道）；
    // 确实要钉死某条族时，直接填数字。
    static constexpr std::uint32_t 自动队列族 = 0xFFFFFFFFu;

    std::uint32_t 队列族索引 = 自动队列族;
    std::uint32_t 队列序号   = 0;      // 同一族内取第几条队列
    int 线程数 = 0;                    // 0 = 该阶段不启用；>0 = 该阶段命令流/工作线程数
};

// ---------------------------------------------------------------------------
// 逻辑设备注册参数：整份交给 vk底层::注册逻辑设备()，本类不碰枚举与创建
// ---------------------------------------------------------------------------
struct 设备注册信息 {
    std::string 设备名;                  // 空 = 按 物理设备索引 选
    std::uint32_t 物理设备索引 = 0;
    std::vector<std::string> 实例扩展;   // 额外要开的设备扩展
    std::string 应用名 = "vk张量系统";
    bool 允许软件实现 = false;           // 找不到硬件时是否接受 software 设备
};

// ---------------------------------------------------------------------------
// 派发规格：显式指定 vkCmdDispatch 的三维组数，用于非标准 1D 派发的算子
// ---------------------------------------------------------------------------
struct 派发规格 {
    std::uint32_t 组数X = 1;
    std::uint32_t 组数Y = 1;
    std::uint32_t 组数Z = 1;

    // 组数X = 0 是标记值：录制时按首个张量的元素总数与算子 local_size_x 推 1D 组数
    static 派发规格 自动() { return 派发规格{0u, 1u, 1u}; }
    bool 是自动() const { return 组数X == 0u; }
};

// ---------------------------------------------------------------------------
// 着色器元信息：由 着色编译器 生成的 <算子名>.spv.hpp 静态登记
// ---------------------------------------------------------------------------
struct 着色器元信息 {
    std::string   算子名;              // "add" / "sub" / "mul" / "div"
    std::uint32_t 工作组大小 = 64;     // 与 .spv 内 local_size 一致
    std::uint32_t 绑定数     = 0;      // 描述符集个数（每个集一个 binding，下标即 binding）
    std::uint32_t 推送常量字节 = 0;    // push block 尺寸，已对齐到 4
    std::uint32_t 产物指纹   = 0;      // SPIR-V FNV-1a，0 表示跳过校验
};

// ---------------------------------------------------------------------------
// 张量句柄：vk张量 交给调度器的最小信息，持有底层 shared_ptr 保证在途安全
// ---------------------------------------------------------------------------
struct 张量句柄 {
    std::shared_ptr<vk底层::缓冲>      缓冲;
    std::shared_ptr<vk底层::描述符集>  描述符集;
    std::uint32_t 元素总数 = 0;
    std::uint32_t 元素字节 = 0;

    bool 有效() const { return 缓冲 && 描述符集 && 元素总数 > 0 && 元素字节 > 0; }
};

// ---------------------------------------------------------------------------
// 算子步骤：多核 pass 里的一格，对应一条命令缓冲内的一次 dispatch
// ---------------------------------------------------------------------------
struct 算子步骤 {
    std::string                算子名;
    std::vector<张量句柄>      张量列表;
    std::vector<std::uint8_t>  推送常量;                       // 长度须等于该算子的 推送常量字节
    派发规格                   派发 = 派发规格::自动();
    // 同一条命令缓冲里相邻 dispatch 之间，驱动不保证前一次的写对后一次可见
    // （实测 AMD RADV 会读到旧数据），所以默认在每步前插一道 compute 屏障。
    // 本步不读前面任何步骤的输出时，置 true 省掉这道屏障。第一步永远不插。
    bool                       不需前置屏障 = false;
};

// 从任意推送结构推出一份算子步骤（结构体自己按 std430 对齐好，这里只按字节搬）
template <class 推送类型>
inline 算子步骤 构造算子步骤(const std::string& 算子名,
                             std::vector<张量句柄> 张量列表,
                             const 推送类型& 推送,
                             派发规格 派发 = 派发规格::自动()) {
    static_assert(sizeof(推送类型) % 4u == 0u, "推送结构必须 4 字节对齐");
    算子步骤 步骤;
    步骤.算子名   = 算子名;
    步骤.张量列表 = std::move(张量列表);
    步骤.推送常量.resize(sizeof(推送类型));
    std::memcpy(步骤.推送常量.data(), &推送, sizeof(推送类型));
    步骤.派发     = 派发;
    return 步骤;
}

// 无推送常量的算子（.spv 元信息 推送常量字节 = 0）
inline 算子步骤 构造算子步骤(const std::string& 算子名,
                             std::vector<张量句柄> 张量列表,
                             派发规格 派发 = 派发规格::自动()) {
    算子步骤 步骤;
    步骤.算子名   = 算子名;
    步骤.张量列表 = std::move(张量列表);
    步骤.派发     = 派发;
    return 步骤;
}

// ---------------------------------------------------------------------------
// 标准算子推送常量：与 vk张量::算子推送常量 逐字节一致
// ---------------------------------------------------------------------------
struct 基础算子常量 {
    std::uint32_t 元素总数 = 0;
    std::uint32_t 保留[3]  = {0, 0, 0};
};
static_assert(sizeof(基础算子常量) == 16, "推送常量布局必须与 .spv 的 push block 一致");

// ---------------------------------------------------------------------------
// 流水线任务
// ---------------------------------------------------------------------------
struct 流水线任务 {
    std::uint64_t 编号 = 0;
    流水线阶段    阶段 = 流水线阶段::图形;

    // 录制回调：命令流已进入录制态，回调内只需记录命令。
    std::function<void(VkCommandBuffer)> 录制;

    std::vector<std::uint64_t> 前置依赖;
    std::vector<std::uint64_t> 后继任务;

    std::atomic<std::uint32_t> 未完成依赖数{0};
    std::atomic<任务状态>      状态{任务状态::等待};
    int                        优先级 = 0;
    // 录制期异常的原因文本，由工作线程在 标记结束 前写入（受任务表锁保护）
    std::string                失败原因;
};

// 优先队列比较器：优先级大者先出；同级按编号，保证 FCFS
struct 任务优先级比较 {
    bool operator()(const std::shared_ptr<流水线任务>& 左,
                    const std::shared_ptr<流水线任务>& 右) const {
        if (左->优先级 != 右->优先级) return 左->优先级 < 右->优先级;
        return 左->编号 > 右->编号;
    }
};

// ---------------------------------------------------------------------------
// 全局流水线调度器
// ---------------------------------------------------------------------------
class 流水线调度器 {
public:
    static 流水线调度器& 获取();

    流水线调度器(const 流水线调度器&) = delete;
    流水线调度器& operator=(const 流水线调度器&) = delete;

    // 启动（默认路径）：让 vk底层 注册逻辑设备，注册结果由本类持有，
    // 停止() 时释放。需要哪些队列用途按 配置 里 线程数 > 0 的阶段汇总上去。
    // 着色器目录非空时顺带加载磁盘 .spv（元信息缺失即抛出，不静默退默认布局）。
    void 启动(const 设备注册信息& 注册, const std::vector<阶段配置>& 配置,
              const std::string& 着色器目录 = "");

    // 启动（已在外部注册好设备）：交出 shared_ptr，所有权同样归本类。
    // 传 nullptr 或空指针一律抛错；本重载存在的意义是让 vk底层 之外的
    // 特殊创建流程（多设备、共享 device）也能接进同一套调度。
    void 启动(std::shared_ptr<vk底层::设备> 设备, const std::vector<阶段配置>& 配置,
              const std::string& 着色器目录 = "");

    // 停止：清空队列、标记非终态任务为取消、join 线程、释放命令流与管线，
    // 最后放掉持有的逻辑设备。
    void 停止();

    bool 运行中() const { return 运行标志.load(std::memory_order_acquire); }

    // 提交任务，返回编号。依赖必须已存在，否则抛异常。
    std::uint64_t 提交(
        流水线阶段 阶段,
        std::function<void(VkCommandBuffer)> 录制,
        const std::vector<std::uint64_t>& 依赖 = {},
        int 优先级 = 0);

    void       等待(std::uint64_t 编号);
    void       等待全部();
    // 清空已完成任务的录制闭包：闭包按值捕获张量句柄（shared_ptr），不清空时
    // 一整轮前向的逐层权重与中间量会在任务表里常驻到服务退出。任务编号与终态
    // 保留，后续按编号查依赖的行为不变。
    void       释放已完成录制();
    任务状态   查询状态(std::uint64_t 编号) const;
    bool       已完成(std::uint64_t 编号) const;
    // 任务终态为 失败 时取原因；录制期抛出的异常文本会落到这里
    std::string 查询失败原因(std::uint64_t 编号) const;
    std::size_t 队列积压(流水线阶段 阶段) const;
    std::size_t 任务表大小() const;

    // 借用的设备引用：所有权在调度器，vk张量 只拿引用建缓冲，别存指针。
    vk底层::设备& 获取设备() const;
    bool 持有设备() const { return 设备_ != nullptr; }

    // ---- 二进制 .spv 加载 ----
    // 生成的 <算子名>.spv.hpp 在静态初始化期调用
    void 登记元信息(着色器元信息 信息);
    // 扫目录：文件名去扩展名即算子名，逐个读二进制、校验、用持有的设备建管线
    std::size_t 加载着色器(const std::string& 目录, bool 递归 = true, bool 严格 = true);
    const 着色器元信息* 查询元信息(const std::string& 算子名) const;
    bool 存在算子(const std::string& 算子名) const;
    std::size_t 算子数量() const;
    std::vector<std::string> 已加载算子() const;
    std::vector<std::string> 加载失败清单() const;

    // ---- 布局封装：vk张量 只交张量，命令流、绑定与派发全部在这里完成 ----
    // 调用方永远不传 VkCommandBuffer：命令流由调度器出具，录制由调度器执行。
    // 同步录制：借调度器的计算命令流，录制完立即提交并等围栏，函数返回即已执行。
    void 录制基础算子(const std::string& 算子名,
                      const std::vector<张量句柄>& 张量列表,
                      const 基础算子常量& 额外常量 = {});
    // 自定义推送常量入口：算子的 push block 不是标准 16 字节时用
    void 录制自定义推送算子(const std::string& 算子名,
                            const std::vector<张量句柄>& 张量列表,
                            const void* 推送数据, std::uint32_t 推送字节);
    // 自定义派发入口：显式指定 vkCmdDispatch 的三维组数
    void 录制自定义派发算子(const std::string& 算子名,
                            const std::vector<张量句柄>& 张量列表,
                            const void* 推送数据, std::uint32_t 推送字节,
                            派发规格 派发);
    // 一步式：排进计算阶段队列，返回任务编号，命令流由工作线程自己的那条出具
    std::uint64_t 提交基础算子(const std::string& 算子名,
                               std::vector<张量句柄> 张量列表,
                               const std::vector<std::uint64_t>& 依赖 = {},
                               int 优先级 = 0);
    // 自定义派发：显式指定 vkCmdDispatch 的三维组数，用于非标准 1D 派发的算子
    std::uint64_t 提交自定义算子(const std::string& 算子名,
                                 std::vector<张量句柄> 张量列表,
                                 const void* 推送数据, std::uint32_t 推送字节,
                                 派发规格 派发,
                                 const std::vector<std::uint64_t>& 依赖 = {},
                                 int 优先级 = 0);

    // ---- 多核 pass：一串算子录进同一条命令缓冲，只提交一次、只等一个围栏 ----
    // 步骤间的读写顺序由 pass 内部自管（默认每步前一道 compute 屏障），
    // 不用再挂任务依赖；跨 pass 才需要 依赖。
    // 每一步的 绑定数/推送字节 仍按各自 .spv 元信息校验，在调用线程抛出。
    void 录制算子串(const std::vector<算子步骤>& 步骤列表);
    std::uint64_t 提交算子串(std::vector<算子步骤> 步骤列表,
                             const std::vector<std::uint64_t>& 依赖 = {},
                             int 优先级 = 0);

private:
    // 每条已加载算子：元信息 + 底层管线
    struct 算子条目 {
        着色器元信息 信息;
        std::shared_ptr<vk底层::管线> 管线;
    };

    流水线调度器();
    ~流水线调度器();

    // 每个工作线程独占一个命令流
    struct 线程上下文 {
        std::thread                     线程;
        std::shared_ptr<vk底层::命令流> 命令流;
    };

    using 阶段队列类型 = std::priority_queue<
        std::shared_ptr<流水线任务>,
        std::vector<std::shared_ptr<流水线任务>>,
        任务优先级比较>;

    void 工作线程循环(流水线阶段 阶段, 线程上下文* 上下文);
    void 入队(std::shared_ptr<流水线任务> 任务);
    void 标记结束(std::shared_ptr<流水线任务> 任务, 任务状态 终态);

    // 提交期与录制期共用的参数检查：在调用线程抛错，避免错误被工作线程吞掉
    算子条目 取出条目(const std::string& 算子名) const;
    void 校验录制参数(const std::string& 算子名,
                      const std::vector<张量句柄>& 张量列表,
                      std::uint32_t 推送字节) const;
    // 多核 pass 的逐步校验：任一步不合法即在调用线程抛出
    void 校验算子串(const std::vector<算子步骤>& 步骤列表) const;

    // 真正的录制实现：命令缓冲只在私有层出现，
    // 由 命令流->执行() 交给回调，或由工作线程的命令流给出。
    void 录制到命令流(VkCommandBuffer 命令缓冲, const std::string& 算子名,
                      const std::vector<张量句柄>& 张量列表,
                      const 基础算子常量& 额外常量);
    void 录制自定义到命令流(VkCommandBuffer 命令缓冲, const std::string& 算子名,
                            const std::vector<张量句柄>& 张量列表,
                            const void* 推送数据, std::uint32_t 推送字节);
    // 带自定义派发的录制：显式指定 vkCmdDispatch 的三维组数
    void 录制自定义派发(VkCommandBuffer 命令缓冲, const std::string& 算子名,
                        const std::vector<张量句柄>& 张量列表,
                        const void* 推送数据, std::uint32_t 推送字节,
                        派发规格 派发);
    // 多核 pass 的录制：逐步 bind + push + dispatch，全在同一条命令缓冲里
    void 录制算子串到命令流(VkCommandBuffer 命令缓冲,
                            const std::vector<算子步骤>& 步骤列表);
    // 同步录制用的命令流：首次使用时按计算阶段的队列创建，之后复用。
    // 调用前必须已持有 同步命令流锁_（一条命令流不能并发录制）。
    vk底层::命令流& 取得同步命令流();

    // 阶段 → 队列用途位；用途到队列族/队列的解析在下面三个适配点里
    static VkQueueFlags 阶段用途(流水线阶段 阶段);

    // 二进制 .spv 读取与校验
    static bool 读spirv(const std::filesystem::path& 路径,
                        std::vector<std::uint32_t>& 输出, std::string& 原因);
    static std::uint32_t 计算指纹(const std::vector<std::uint32_t>& 字);

    // ---- vk底层 适配点（名字或参数不同只改这四个函数）----
    // ①注册逻辑设备：物理设备挑选、队列族开启、vkCreateDevice 全在 vk底层
    static std::shared_ptr<vk底层::设备> 注册设备(const 设备注册信息& 注册,
                                                   VkQueueFlags 队列用途);
    // ②按阶段用途问 vk底层 要队列族索引（物理设备属性只有它清楚）
    static std::uint32_t 解析队列族(vk底层::设备& 设备, 流水线阶段 阶段);
    // ③按族索引与序号取一条 VkQueue
    static VkQueue 取队列(vk底层::设备& 设备, std::uint32_t 族索引, std::uint32_t 序号);
    // ④建管线：绑定数 = 描述符集个数，每集一个 binding
    static std::shared_ptr<vk底层::管线> 内建管线(vk底层::设备& 设备,
                                                   const std::vector<std::uint32_t>& 字,
                                                   const 着色器元信息& 信息);

    static constexpr int 阶段总数 = static_cast<int>(流水线阶段::阶段数量);

    // 持有的逻辑设备：启动() 建立所有权，停止() 释放
    std::shared_ptr<vk底层::设备> 设备_;
    std::atomic<bool> 运行标志{false};

    // 计算阶段的队列信息，启动时记下 —— 同步录制命令流用它创建
    std::uint32_t 计算队列族索引_ = 0;
    VkQueue       计算队列_       = VK_NULL_HANDLE;
    std::shared_ptr<vk底层::命令流> 同步命令流_;
    std::mutex    同步命令流锁_;

    std::vector<std::unique_ptr<线程上下文>> 线程池;
    std::vector<int>                        阶段线程数;   // 提交时校验

    std::vector<std::unique_ptr<阶段队列类型>> 阶段队列;
    // 队列积压() 是 const 查询，锁必须 mutable；mutex 不可移动所以用 deque
    mutable std::deque<std::mutex>             阶段队列锁;
    std::deque<std::condition_variable>        阶段队列条件;

    mutable std::mutex 任务表锁;
    std::unordered_map<std::uint64_t, std::shared_ptr<流水线任务>> 任务表;
    std::atomic<std::uint64_t> 下一编号{1};

    mutable std::mutex      完成锁;
    std::condition_variable 完成条件;

    // ---- 着色器侧状态 ----
    mutable std::mutex 算子锁;
    std::unordered_map<std::string, 着色器元信息> 元信息表;
    std::unordered_map<std::string, 算子条目>     算子表;
    std::vector<std::string> 加载失败_;
};

// ============================================================================
// 实现
// ============================================================================

inline 流水线调度器& 流水线调度器::获取() {
    static 流水线调度器 实例;
    return 实例;
}

inline 流水线调度器::流水线调度器() = default;

inline 流水线调度器::~流水线调度器() { 停止(); }

inline vk底层::设备& 流水线调度器::获取设备() const {
    if (设备_ == nullptr) {
        throw std::runtime_error("流水线调度器没有持有设备：先 启动()，"
                                 "或已被 停止() 释放");
    }
    return *设备_;
}

inline VkQueueFlags 流水线调度器::阶段用途(流水线阶段 阶段) {
    switch (阶段) {
    case 流水线阶段::图形: return VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    case 流水线阶段::计算: return VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    case 流水线阶段::传输: return VK_QUEUE_TRANSFER_BIT;
    case 流水线阶段::阶段数量: break;
    }
    return VK_QUEUE_TRANSFER_BIT;
}

// vk底层 适配点①：注册逻辑设备。入口名/参数不同只改这一行。
inline std::shared_ptr<vk底层::设备>
流水线调度器::注册设备(const 设备注册信息& 注册, VkQueueFlags 队列用途) {
    std::vector<const char*> 扩展;
    扩展.reserve(注册.实例扩展.size());
    for (const std::string& 项 : 注册.实例扩展) 扩展.push_back(项.c_str());
    return vk底层::注册逻辑设备(注册.设备名, 注册.物理设备索引, 队列用途, 扩展,
                                注册.应用名, 注册.允许软件实现);
}

// vk底层 适配点②：用途 → 队列族索引。
inline std::uint32_t 流水线调度器::解析队列族(vk底层::设备& 设备, 流水线阶段 阶段) {
    return 设备.获取队列族(阶段用途(阶段));
}

// vk底层 适配点③：族索引 + 序号 → VkQueue。
inline VkQueue 流水线调度器::取队列(vk底层::设备& 设备, std::uint32_t 族索引,
                                    std::uint32_t 序号) {
    return 设备.获取队列(族索引, 序号);
}

inline void 流水线调度器::启动(const 设备注册信息& 注册,
                              const std::vector<阶段配置>& 配置,
                              const std::string& 着色器目录) {
    // 开了几条线程就要几条队列：把用途汇总上去，一次注册到位
    VkQueueFlags 用途 = 0;
    const std::size_t 项数 = 配置.size() < static_cast<std::size_t>(阶段总数)
                                ? 配置.size() : static_cast<std::size_t>(阶段总数);
    for (std::size_t i = 0; i < 项数; ++i) {
        if (配置[i].线程数 > 0) 用途 |= 阶段用途(static_cast<流水线阶段>(i));
    }
    std::shared_ptr<vk底层::设备> 设备;
    try {
        设备 = 注册设备(注册, 用途);
    } catch (const std::exception& 异常) {
        throw std::runtime_error(std::string("注册逻辑设备失败：") + 异常.what());
    }
    启动(std::move(设备), 配置, 着色器目录);
}

inline void 流水线调度器::启动(
    std::shared_ptr<vk底层::设备> 设备, const std::vector<阶段配置>& 配置,
    const std::string& 着色器目录)
{
    if (运行标志.exchange(true, std::memory_order_acq_rel)) return;
    try {
        if (static_cast<int>(配置.size()) != 阶段总数) {
            throw std::runtime_error("阶段配置数量与流水线阶段数量不一致");
        }
        if (!设备) throw std::runtime_error("启动 流水线调度器 需要一个有效的逻辑设备");
        设备_ = std::move(设备);   // 所有权从这里开始归调度器

        阶段队列.resize(阶段总数);
        阶段队列锁.resize(阶段总数);
        阶段队列条件.resize(阶段总数);
        阶段线程数.assign(阶段总数, 0);
        for (int i = 0; i < 阶段总数; ++i) {
            阶段队列[i] = std::make_unique<阶段队列类型>();
        }

        for (int 阶段索引 = 0; 阶段索引 < 阶段总数; ++阶段索引) {
            const 阶段配置& 项 = 配置[阶段索引];
            流水线阶段 阶段值 = static_cast<流水线阶段>(阶段索引);
            if (项.线程数 <= 0) continue;   // 该阶段不启用，不起线程

            std::uint32_t 族索引 = 项.队列族索引;
            if (族索引 == 阶段配置::自动队列族) {
                族索引 = 解析队列族(*设备_, 阶段值);
            }
            VkQueue 队列 = 取队列(*设备_, 族索引, 项.队列序号);
            if (队列 == VK_NULL_HANDLE) {
                throw std::runtime_error(std::string("阶段 ") + 流水线阶段名(阶段值)
                                         + " 取不到队列（队列族 " + std::to_string(族索引)
                                         + " 序号 " + std::to_string(项.队列序号)
                                         + "）——注册逻辑设备时没开这个用途的队列");
            }
            // 同步录制命令流按计算阶段的队列建，这里先把队列记住
            if (阶段值 == 流水线阶段::计算) {
                计算队列族索引_ = 族索引;
                计算队列_       = 队列;
            }
            阶段线程数[阶段索引] = 项.线程数;

            for (int t = 0; t < 项.线程数; ++t) {
                auto 上下文 = std::make_unique<线程上下文>();
                上下文->命令流 = 设备_->创建命令流(族索引, 队列);
                if (!上下文->命令流) {
                    throw std::runtime_error(
                        std::string("为阶段 ") + 流水线阶段名(阶段值)
                        + " 创建命令流失败");
                }
                线程上下文* 原始指针 = 上下文.get();
                原始指针->线程 = std::thread(
                    [this, 阶段值, 原始指针] { 工作线程循环(阶段值, 原始指针); });
                线程池.push_back(std::move(上下文));
            }
        }

        // 一切算子都排进计算阶段，没有它就没有能用的命令流
        if (计算队列_ == VK_NULL_HANDLE) {
            throw std::runtime_error("启动 流水线调度器 至少要启用计算阶段"
                                     "（阶段配置[计算].线程数 > 0）");
        }

        // 着色器加载在起线程之后：失败会走 catch 统一 停止() 回收线程
        if (!着色器目录.empty()) 加载着色器(着色器目录);
    } catch (...) {
        停止();
        throw;
    }
}

inline void 流水线调度器::停止() {
    // 允许在未启动状态下调用做兜底清理
    const bool 曾在运行 = 运行标志.exchange(false, std::memory_order_acq_rel);

    if (曾在运行) {
        // 1) 清空各阶段队列（仍在队列中的任务直接标记取消）
        for (int i = 0; i < static_cast<int>(阶段队列.size()); ++i) {
            std::lock_guard<std::mutex> 守卫(阶段队列锁[i]);
            auto& 队列 = *阶段队列[i];
            while (!队列.empty()) {
                队列.top()->状态.store(任务状态::取消, std::memory_order_release);
                队列.pop();
            }
        }

        // 2) 任务表内非终态任务一律取消
        {
            std::lock_guard<std::mutex> 守卫(任务表锁);
            for (auto& 项 : 任务表) {
                auto 状态 = 项.second->状态.load(std::memory_order_acquire);
                if (状态 != 任务状态::完成 &&
                    状态 != 任务状态::失败 &&
                    状态 != 任务状态::取消) {
                    项.second->状态.store(任务状态::取消, std::memory_order_release);
                }
            }
        }

        // 3) 唤醒并回收工作线程（命令流内部围栏等待会自然返回）
        for (auto& 条件 : 阶段队列条件) 条件.notify_all();
        完成条件.notify_all();
        for (auto& 上下文 : 线程池) {
            if (上下文->线程.joinable()) 上下文->线程.join();
        }

        // 4) 线程全部退出后释放命令流（vk底层 RAII 负责命令池与围栏销毁）
        线程池.clear();
        {
            std::lock_guard<std::mutex> 守卫(同步命令流锁_);
            同步命令流_.reset();          // 设备即将失效，同步流一起放掉
        }
        计算队列_ = VK_NULL_HANDLE;
        计算队列族索引_ = 0;
        阶段线程数.clear();
        阶段队列.clear();
        阶段队列锁.clear();
        阶段队列条件.clear();
        {
            std::lock_guard<std::mutex> 守卫(任务表锁);
            任务表.clear();
        }
    }

    // 5) 管线必须在所有录制线程退出之后再销毁，否则在途命令引用已销毁对象
    {
        std::lock_guard<std::mutex> 守卫(算子锁);
        算子表.clear();
        加载失败_.clear();
    }

    // 6) 最后才放设备：前面所有管线/命令流都毁了，vk底层::设备 析构才安全
    设备_.reset();
}

inline std::uint64_t 流水线调度器::提交(
    流水线阶段 阶段,
    std::function<void(VkCommandBuffer)> 录制,
    const std::vector<std::uint64_t>& 依赖,
    int 优先级)
{
    if (!运行标志.load(std::memory_order_acquire)) {
        throw std::runtime_error("流水线调度器未启动或已停止，无法提交任务");
    }
    int 索引 = static_cast<int>(阶段);
    if (索引 < 0 || 索引 >= 阶段总数 || 阶段队列[索引] == nullptr) {
        throw std::runtime_error("目标阶段未初始化");
    }
    if (阶段线程数[索引] == 0) {
        throw std::runtime_error(
            std::string("目标阶段无工作线程: ") + 流水线阶段名(阶段));
    }

    auto 任务      = std::make_shared<流水线任务>();
    任务->阶段     = 阶段;
    任务->录制     = std::move(录制);
    任务->优先级   = 优先级;

    std::uint32_t 实际未完成 = 0;
    {
        std::lock_guard<std::mutex> 守卫(任务表锁);

        for (std::uint64_t 前置编号 : 依赖) {
            if (任务表.find(前置编号) == 任务表.end()) {
                throw std::runtime_error(
                    "前置依赖任务不存在: " + std::to_string(前置编号));
            }
        }

        任务->编号 = 下一编号.fetch_add(1, std::memory_order_relaxed);
        任务表.emplace(任务->编号, 任务);

        // 关键：状态检查与后继登记必须同锁，避免与"完成方读取后继列表"竞态
        for (std::uint64_t 前置编号 : 依赖) {
            auto 前置 = 任务表[前置编号];
            auto 前置状态 = 前置->状态.load(std::memory_order_acquire);
            if (前置状态 == 任务状态::完成 ||
                前置状态 == 任务状态::失败 ||
                前置状态 == 任务状态::取消) {
                continue;
            }
            前置->后继任务.push_back(任务->编号);
            ++实际未完成;
        }
        任务->未完成依赖数.store(实际未完成, std::memory_order_release);
        任务->前置依赖 = 依赖;
    }

    if (任务->未完成依赖数.load(std::memory_order_acquire) == 0) {
        任务->状态.store(任务状态::就绪, std::memory_order_release);
        入队(任务);
    }
    return 任务->编号;
}

inline void 流水线调度器::等待(std::uint64_t 编号) {
    std::unique_lock<std::mutex> 守卫(完成锁);
    完成条件.wait(守卫, [this, 编号] {
        std::lock_guard<std::mutex> 表锁(任务表锁);
        auto 迭代 = 任务表.find(编号);
        if (迭代 == 任务表.end()) return true;
        auto 状态 = 迭代->second->状态.load(std::memory_order_acquire);
        return 状态 == 任务状态::完成 ||
               状态 == 任务状态::失败 ||
               状态 == 任务状态::取消;
    });
}

inline void 流水线调度器::等待全部() {
    std::unique_lock<std::mutex> 守卫(完成锁);
    完成条件.wait(守卫, [this] {
        std::lock_guard<std::mutex> 表锁(任务表锁);
        for (auto& 项 : 任务表) {
            auto 状态 = 项.second->状态.load(std::memory_order_acquire);
            if (状态 != 任务状态::完成 &&
                状态 != 任务状态::失败 &&
                状态 != 任务状态::取消) return false;
        }
        return true;
    });
}

inline void 流水线调度器::释放已完成录制() {
    std::lock_guard<std::mutex> 守卫(任务表锁);
    for (auto& 项 : 任务表) {
        if (项.second->状态.load(std::memory_order_acquire) == 任务状态::完成)
            项.second->录制 = nullptr;
    }
}

inline 任务状态 流水线调度器::查询状态(std::uint64_t 编号) const {
    std::lock_guard<std::mutex> 守卫(任务表锁);
    auto 迭代 = 任务表.find(编号);
    if (迭代 == 任务表.end()) return 任务状态::取消;
    return 迭代->second->状态.load(std::memory_order_acquire);
}

inline bool 流水线调度器::已完成(std::uint64_t 编号) const {
    auto 状态 = 查询状态(编号);
    return 状态 == 任务状态::完成 ||
           状态 == 任务状态::失败 ||
           状态 == 任务状态::取消;
}

inline std::string 流水线调度器::查询失败原因(std::uint64_t 编号) const {
    std::lock_guard<std::mutex> 守卫(任务表锁);
    auto 迭代 = 任务表.find(编号);
    if (迭代 == 任务表.end()) return std::string();
    return 迭代->second->失败原因;
}

inline std::size_t 流水线调度器::队列积压(流水线阶段 阶段) const {
    int 索引 = static_cast<int>(阶段);
    std::lock_guard<std::mutex> 守卫(阶段队列锁[索引]);
    return 阶段队列[索引]->size();
}

inline std::size_t 流水线调度器::任务表大小() const {
    std::lock_guard<std::mutex> 守卫(任务表锁);
    return 任务表.size();
}

// ---------------------------------------------------------------------------
// 内部：工作线程主循环
//   录制/提交/等待全部委托给 命令流->执行()，调度器不碰 Vulkan 资源接口。
// ---------------------------------------------------------------------------
inline void 流水线调度器::工作线程循环(
    流水线阶段 阶段, 线程上下文* 上下文)
{
    int 索引 = static_cast<int>(阶段);
    auto& 队列 = *阶段队列[索引];
    auto& 锁   = 阶段队列锁[索引];
    auto& 条件 = 阶段队列条件[索引];

    while (true) {
        std::shared_ptr<流水线任务> 任务;
        {
            std::unique_lock<std::mutex> 守卫(锁);
            条件.wait(守卫, [&] {
                return !运行标志.load(std::memory_order_acquire) || !队列.empty();
            });
            if (!运行标志.load(std::memory_order_acquire)) return;
            任务 = 队列.top();
            队列.pop();
        }

        // 已取消（失败级联或停止）则跳过
        if (任务->状态.load(std::memory_order_acquire) != 任务状态::就绪) {
            完成条件.notify_all();
            continue;
        }
        任务->状态.store(任务状态::执行中, std::memory_order_release);

        任务状态 终态 = 任务状态::完成;
        std::string 原因;
        try {
            // 命令流内部完成：分配 → 开始 → 录制 → 结束 → 提交 → 等围栏 → 释放
            上下文->命令流->执行(任务->录制);
        } catch (const std::exception& 异常) {
            终态 = 任务状态::失败;
            原因 = 异常.what();
        } catch (...) {
            终态 = 任务状态::失败;
            原因 = "录制回调抛出了非 std::exception 异常";
        }
        // 执行() 已等过围栏，GPU 读完缓冲；录制闭包捕获的张量句柄（shared_ptr<缓冲>）
        // 此后不再需要。立刻放掉，否则长循环里每个已完成任务的权重缓冲会一直挂在任务表上，
        // 累积撑爆设备堆（VK_ERROR_OUT_OF_DEVICE_MEMORY）。任务本身留在表里供状态查询。
        任务->录制 = nullptr;
        if (终态 == 任务状态::失败) {
            std::lock_guard<std::mutex> 守卫(任务表锁);
            任务->失败原因 = 原因;
        }

        标记结束(任务, 终态);
    }
}

// ---------------------------------------------------------------------------
// 内部：入队
// ---------------------------------------------------------------------------
inline void 流水线调度器::入队(std::shared_ptr<流水线任务> 任务) {
    int 索引 = static_cast<int>(任务->阶段);
    {
        std::lock_guard<std::mutex> 守卫(阶段队列锁[索引]);
        阶段队列[索引]->push(任务);
    }
    阶段队列条件[索引].notify_one();
}

// ---------------------------------------------------------------------------
// 内部：标记任务结束并推进后继
// ---------------------------------------------------------------------------
inline void 流水线调度器::标记结束(
    std::shared_ptr<流水线任务> 任务, 任务状态 终态)
{
    std::vector<std::shared_ptr<流水线任务>> 就绪列表;

    {
        std::lock_guard<std::mutex> 守卫(任务表锁);

        // 状态写入与读取后继列表同锁，避免与提交时的"状态检查+登记后继"竞态
        任务->状态.store(终态, std::memory_order_release);

        if (终态 == 任务状态::完成) {
            for (std::uint64_t 后继编号 : 任务->后继任务) {
                auto 迭代 = 任务表.find(后继编号);
                if (迭代 == 任务表.end()) continue;
                auto 后继 = 迭代->second;

                任务状态 期望 = 任务状态::等待;
                if (后继->状态.load(std::memory_order_acquire) != 期望) continue;

                std::uint32_t 旧值 = 后继->未完成依赖数.fetch_sub(
                    1, std::memory_order_acq_rel);
                if (旧值 != 1) continue;

                if (后继->状态.compare_exchange_strong(
                        期望, 任务状态::就绪, std::memory_order_acq_rel)) {
                    就绪列表.push_back(后继);
                }
            }
        } else {
            // 失败/取消：对仍处于"等待"的后继做级联取消
            std::vector<std::uint64_t> 栈(
                任务->后继任务.begin(), 任务->后继任务.end());
            while (!栈.empty()) {
                std::uint64_t 编号 = 栈.back();
                栈.pop_back();

                auto 迭代 = 任务表.find(编号);
                if (迭代 == 任务表.end()) continue;
                auto 候选 = 迭代->second;

                任务状态 期望 = 任务状态::等待;
                if (!候选->状态.compare_exchange_strong(
                        期望, 任务状态::取消, std::memory_order_acq_rel)) {
                    continue;
                }
                栈.insert(栈.end(),
                          候选->后继任务.begin(), 候选->后继任务.end());
            }
        }
    }

    // 锁外入队，避免持任务表锁做队列操作
    for (auto& 就绪 : 就绪列表) 入队(就绪);
    完成条件.notify_all();
}

// ---------------------------------------------------------------------------
// 二进制 .spv 读取
// ---------------------------------------------------------------------------
inline std::uint32_t 流水线调度器::计算指纹(const std::vector<std::uint32_t>& 字) {
    std::uint32_t 指纹 = 2166136261u;                 // FNV-1a 32 位
    for (std::uint32_t 字值 : 字) { 指纹 ^= 字值; 指纹 *= 16777619u; }
    return 指纹;
}

inline bool 流水线调度器::读spirv(const std::filesystem::path& 路径,
                                  std::vector<std::uint32_t>& 输出,
                                  std::string& 原因) {
    std::error_code 错误;
    auto 大小 = std::filesystem::file_size(路径, 错误);
    if (错误) { 原因 = "取文件大小失败: " + 错误.message(); return false; }
    if (大小 == 0) { 原因 = "文件为空"; return false; }
    if (大小 % 4 != 0) { 原因 = "长度不是 4 的倍数，不是 SPIR-V 字流"; return false; }

    std::ifstream 流(路径, std::ios::binary);
    if (!流) { 原因 = "打开失败"; return false; }

    std::vector<char> 字节(static_cast<std::size_t>(大小));
    流.read(字节.data(), static_cast<std::streamsize>(大小));
    if (!流) { 原因 = "读取不完整"; return false; }

    输出.resize(static_cast<std::size_t>(大小 / 4));
    std::memcpy(输出.data(), 字节.data(), static_cast<std::size_t>(大小));

    // SPIR-V 魔数：小端 0x07230203，大端按字节序翻转
    if (输出[0] != 0x07230203u && 输出[0] != 0x03022307u) {
        原因 = "SPIR-V 魔数不符——很可能是没编译的 GLSL 源直接命名成了 .spv";
        输出.clear();
        return false;
    }
    return true;
}

// 唯一需要跟 vk底层 对齐的函数：入口名或参数不同就只改这里。
// 约定：绑定数 = 描述符集个数，每集一个 binding（与 录制基础算子 的下标绑定一致）。
inline std::shared_ptr<vk底层::管线>
流水线调度器::内建管线(vk底层::设备& 设备,
                       const std::vector<std::uint32_t>& 字,
                       const 着色器元信息& 信息) {
    return 设备.创建计算管线(字, 信息.绑定数, 信息.工作组大小, 信息.推送常量字节);
}

inline void 流水线调度器::登记元信息(着色器元信息 信息) {
    if (信息.算子名.empty()) return;
    if (信息.推送常量字节 % 4u != 0) {
        std::fprintf(stderr, "[流水线调度器] %s：推送常量 %u 字节不是 4 的倍数，已向上对齐\n",
                     信息.算子名.c_str(), 信息.推送常量字节);
        信息.推送常量字节 += 4u - (信息.推送常量字节 % 4u);
    }
    std::lock_guard<std::mutex> 守卫(算子锁);
    auto 迭代 = 元信息表.find(信息.算子名);
    if (迭代 != 元信息表.end()) {
        // 生成头被多个编译单元包含时静态对象重复构造，内容必须一致
        if (迭代->second.产物指纹 != 信息.产物指纹) {
            std::fprintf(stderr, "[流水线调度器] %s：两份生成头指纹不一致（%u / %u），"
                                 "检查 .spv 与 .spv.hpp 是否同源\n",
                         信息.算子名.c_str(), 迭代->second.产物指纹, 信息.产物指纹);
        }
        return;
    }
    元信息表.emplace(信息.算子名, std::move(信息));
}

inline const 着色器元信息* 流水线调度器::查询元信息(const std::string& 算子名) const {
    std::lock_guard<std::mutex> 守卫(算子锁);
    auto 迭代 = 元信息表.find(算子名);
    if (迭代 == 元信息表.end()) return nullptr;
    return &迭代->second;
}

inline std::size_t 流水线调度器::加载着色器(const std::string& 目录,
                                            bool 递归, bool 严格) {
    vk底层::设备& 设备 = 获取设备();   // 用持有的设备建管线
    std::error_code 错误;
    std::filesystem::path 根(目录);
    if (!std::filesystem::exists(根, 错误) || !std::filesystem::is_directory(根, 错误)) {
        throw std::runtime_error("着色器目录不存在: " + 目录);
    }

    std::vector<std::filesystem::path> 清单;
    auto 收集 = [&](const std::filesystem::path& 路径) {
        if (路径.extension() == ".spv") 清单.push_back(路径);
    };
    if (递归) {
        for (std::filesystem::recursive_directory_iterator 它(根, 错误), 尾;
             它 != 尾; 它.increment(错误)) {
            if (错误) break;
            if (它->is_regular_file()) 收集(它->path());
        }
    } else {
        for (const auto& 项 : std::filesystem::directory_iterator(根, 错误)) {
            if (错误) break;
            if (项.is_regular_file()) 收集(项.path());
        }
    }
    std::sort(清单.begin(), 清单.end());

    std::vector<std::string> 原因清单;
    std::size_t 成功数 = 0;

    for (const auto& 路径 : 清单) {
        const std::string 算子名 = 路径.stem().string();     // add.spv → "add"

        std::vector<std::uint32_t> 字;
        std::string 读取原因;
        if (!读spirv(路径, 字, 读取原因)) {
            原因清单.push_back(路径.string() + ": " + 读取原因);
            continue;
        }

        const 着色器元信息* 信息 = 查询元信息(算子名);
        if (信息 == nullptr) {
            // 找到 .spv 但没有元信息：只警告，不阻塞其他算子加载。
            // 真正使用时会在 取出条目() 里报错。
            std::fprintf(stderr, "[流水线调度器] 跳过 %s: 没有 .spv.hpp 登记元信息\n",
                         算子名.c_str());
            continue;
        }
        if (信息->产物指纹 != 0u) {
            const std::uint32_t 实际 = 计算指纹(字);
            if (实际 != 信息->产物指纹) {
                原因清单.push_back(路径.string() + ": SPIR-V 指纹 " + std::to_string(实际)
                                   + " 与元信息 " + std::to_string(信息->产物指纹)
                                   + " 不符，产物不同源，请重跑 着色编译器");
                continue;
            }
        }

        std::shared_ptr<vk底层::管线> 管线;
        try {
            管线 = 内建管线(设备, 字, *信息);
        } catch (const std::exception& 异常) {
            原因清单.push_back(路径.string() + ": 建管线失败 —— " + 异常.what());
            continue;
        }
        if (!管线) {
            原因清单.push_back(路径.string() + ": vk底层 返回空管线");
            continue;
        }

        {
            std::lock_guard<std::mutex> 守卫(算子锁);
            算子表[算子名] = 算子条目{*信息, std::move(管线)};
        }
        ++成功数;
    }

    {
        std::lock_guard<std::mutex> 守卫(算子锁);
        加载失败_ = 原因清单;
    }

    if (严格 && !原因清单.empty()) {
        std::string 汇总 = "着色器加载失败 " + std::to_string(原因清单.size()) + " 项：\n";
        for (const auto& 项 : 原因清单) 汇总 += "  " + 项 + "\n";
        throw std::runtime_error(汇总);
    }
    return 成功数;
}

inline bool 流水线调度器::存在算子(const std::string& 算子名) const {
    std::lock_guard<std::mutex> 守卫(算子锁);
    return 算子表.find(算子名) != 算子表.end();
}

inline std::size_t 流水线调度器::算子数量() const {
    std::lock_guard<std::mutex> 守卫(算子锁);
    return 算子表.size();
}

inline std::vector<std::string> 流水线调度器::已加载算子() const {
    std::lock_guard<std::mutex> 守卫(算子锁);
    std::vector<std::string> 名字;
    名字.reserve(算子表.size());
    for (const auto& 项 : 算子表) 名字.push_back(项.first);
    std::sort(名字.begin(), 名字.end());
    return 名字;
}

inline std::vector<std::string> 流水线调度器::加载失败清单() const {
    std::lock_guard<std::mutex> 守卫(算子锁);
    return 加载失败_;
}

// ---------------------------------------------------------------------------
// 布局封装录制：绑定、推送常量、派发规模全部按登记的元信息推导
//   约定：第 i 个张量的描述符集绑到 binding i（与 着色编译器 校验的连续编号一致）
// ---------------------------------------------------------------------------
inline 流水线调度器::算子条目 流水线调度器::取出条目(const std::string& 算子名) const {
    std::lock_guard<std::mutex> 守卫(算子锁);
    auto 迭代 = 算子表.find(算子名);
    if (迭代 == 算子表.end()) {
        throw std::runtime_error("算子未加载: " + 算子名
                                 + "（检查 .spv 是否在扫描目录里、.spv.hpp 是否被包含、"
                                 + "以及 启动() 是否传了着色器目录）");
    }
    return 迭代->second;
}

// 与 .spv 布局相关的一切检查都在这里：张量个数对不上绑定数、推送字节对不上、
// 同形状/同类型不满足，都在调用线程抛出，而不是变成一条静默失败的任务。
inline void 流水线调度器::校验录制参数(const std::string& 算子名,
                                       const std::vector<张量句柄>& 张量列表,
                                       std::uint32_t 推送字节) const {
    const 算子条目 条目 = 取出条目(算子名);

    if (张量列表.size() != 条目.信息.绑定数) {
        throw std::runtime_error("算子 " + 算子名 + " 需要 "
                                 + std::to_string(条目.信息.绑定数) + " 个张量，实际 "
                                 + std::to_string(张量列表.size()) + " 个");
    }
    if (推送字节 != 条目.信息.推送常量字节) {
        throw std::runtime_error("算子 " + 算子名 + " 的推送常量应为 "
                                 + std::to_string(条目.信息.推送常量字节) + " 字节，实际 "
                                 + std::to_string(推送字节) + " 字节");
    }

    const std::uint32_t 元素总数 = 张量列表.empty() ? 0u : 张量列表.front().元素总数;
    const std::uint32_t 元素字节 = 张量列表.empty() ? 0u : 张量列表.front().元素字节;

    for (std::size_t i = 0; i < 张量列表.size(); ++i) {
        const 张量句柄& 项 = 张量列表[i];
        if (!项.有效()) {
            throw std::runtime_error("算子 " + 算子名 + " 第 " + std::to_string(i)
                                     + " 个张量未初始化");
        }
        // 注意：不强制要求所有张量同形状，因为自定义算子（如注意力核）的输入可以有不同维度
        // GLSL代码应自行通过推送常量处理不同大小的缓冲区
        // 允许混合精度：fp16 梯度 + fp32 master/m/v 可以在同一 dispatch 里共存；
        // 具体类型由 GLSL 的 binding 声明决定，调度器只负责绑定。
        (void)元素字节;
    }
}

// ---------------------------------------------------------------------------
// 同步录制入口：命令流由调度器出具，调用方一个句柄都不传
// ---------------------------------------------------------------------------
// 取用者必须已持有 同步命令流锁_；一条命令流不能并发录制，所以执行整段都在锁内。
inline vk底层::命令流& 流水线调度器::取得同步命令流() {
    if (!运行中()) throw std::runtime_error("流水线调度器未启动，无法录制");
    if (!同步命令流_) {
        if (!设备_ || 计算队列_ == VK_NULL_HANDLE) {
            throw std::runtime_error("计算阶段没有可用队列（启动 时该阶段没开），"
                                     "无法创建同步录制命令流");
        }
        同步命令流_ = 设备_->创建命令流(计算队列族索引_, 计算队列_);
        if (!同步命令流_) throw std::runtime_error("创建同步录制命令流失败");
    }
    return *同步命令流_;
}

inline void 流水线调度器::录制自定义推送算子(
    const std::string& 算子名, const std::vector<张量句柄>& 张量列表,
    const void* 推送数据, std::uint32_t 推送字节) {
    // 检查先行：参数不对时连命令流都不该被占用
    校验录制参数(算子名, 张量列表, 推送字节);
    std::lock_guard<std::mutex> 守卫(同步命令流锁_);
    取得同步命令流().执行([this, 算子名, &张量列表, 推送数据, 推送字节](VkCommandBuffer 命令缓冲) {
        录制自定义到命令流(命令缓冲, 算子名, 张量列表, 推送数据, 推送字节);
    });
}

inline void 流水线调度器::录制自定义派发算子(
    const std::string& 算子名, const std::vector<张量句柄>& 张量列表,
    const void* 推送数据, std::uint32_t 推送字节,
    派发规格 派发) {
    校验录制参数(算子名, 张量列表, 推送字节);
    std::lock_guard<std::mutex> 守卫(同步命令流锁_);
    取得同步命令流().执行([this, 算子名, &张量列表, 推送数据, 推送字节, 派发](VkCommandBuffer 命令缓冲) {
        录制自定义派发(命令缓冲, 算子名, 张量列表, 推送数据, 推送字节, 派发);
    });
}

inline void 流水线调度器::录制基础算子(const std::string& 算子名,
                                       const std::vector<张量句柄>& 张量列表,
                                       const 基础算子常量& 额外常量) {
    const 着色器元信息* 信息 = 查询元信息(算子名);
    if (信息 == nullptr) {
        throw std::runtime_error("算子未登记元信息: " + 算子名
                                 + "（对应 .spv.hpp 没被任何编译单元包含）");
    }
    if (信息->推送常量字节 != sizeof(基础算子常量)) {
        throw std::runtime_error("算子 " + 算子名 + " 的推送常量是 "
                                 + std::to_string(信息->推送常量字节)
                                 + " 字节，不是标准 16 字节，请改用 录制自定义推送算子");
    }
    校验录制参数(算子名, 张量列表, sizeof(基础算子常量));

    std::lock_guard<std::mutex> 守卫(同步命令流锁_);
    取得同步命令流().执行([this, 算子名, &张量列表, 额外常量](VkCommandBuffer 命令缓冲) {
        录制到命令流(命令缓冲, 算子名, 张量列表, 额外常量);
    });
}

// ---- 私有层：命令缓冲只出现在这两处 ----
inline void 流水线调度器::录制到命令流(VkCommandBuffer 命令缓冲, const std::string& 算子名,
                                       const std::vector<张量句柄>& 张量列表,
                                       const 基础算子常量& 额外常量) {
    基础算子常量 常量 = 额外常量;
    常量.元素总数 = 张量列表.empty() ? 0u : 张量列表.front().元素总数;
    录制自定义到命令流(命令缓冲, 算子名, 张量列表, &常量, sizeof(常量));
}

// 一维自动派发：录制时按元信息把首个张量的元素总数折算成组数
inline void 流水线调度器::录制自定义到命令流(
    VkCommandBuffer 命令缓冲, const std::string& 算子名,
    const std::vector<张量句柄>& 张量列表,
    const void* 推送数据, std::uint32_t 推送字节)
{
    录制自定义派发(命令缓冲, 算子名, 张量列表, 推送数据, 推送字节, 派发规格::自动());
}

inline void 流水线调度器::录制自定义派发(
    VkCommandBuffer 命令缓冲, const std::string& 算子名,
    const std::vector<张量句柄>& 张量列表,
    const void* 推送数据, std::uint32_t 推送字节,
    派发规格 派发)
{
    校验录制参数(算子名, 张量列表, 推送字节);
    const 算子条目 条目 = 取出条目(算子名);

    std::vector<VkDescriptorSet> 集合;
    集合.reserve(张量列表.size());
    for (const 张量句柄& 项 : 张量列表) 集合.push_back(项.描述符集->句柄());

    vkCmdBindPipeline(命令缓冲, VK_PIPELINE_BIND_POINT_COMPUTE, 条目.管线->管线句柄());

    if (!集合.empty()) {
        vkCmdBindDescriptorSets(命令缓冲, VK_PIPELINE_BIND_POINT_COMPUTE,
                                条目.管线->布局句柄(), 0,
                                static_cast<std::uint32_t>(集合.size()),
                                集合.data(), 0, nullptr);
    }

    if (条目.信息.推送常量字节 > 0) {
        vkCmdPushConstants(命令缓冲, 条目.管线->布局句柄(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, 条目.信息.推送常量字节, 推送数据);
    }

    if (派发.是自动()) {
        const std::uint32_t 元素总数 = 张量列表.empty() ? 0u : 张量列表.front().元素总数;
        const std::uint32_t 工作组 = 条目.信息.工作组大小 == 0 ? 1u : 条目.信息.工作组大小;
        const std::uint32_t 组数 = (元素总数 + 工作组 - 1u) / 工作组;
        派发 = 派发规格{组数 == 0 ? 1u : 组数, 1u, 1u};
    }
    vkCmdDispatch(命令缓冲, 派发.组数X, 派发.组数Y, 派发.组数Z);
}

inline std::uint64_t 流水线调度器::提交基础算子(
    const std::string& 算子名, std::vector<张量句柄> 张量列表,
    const std::vector<std::uint64_t>& 依赖, int 优先级) {
    // 调用线程先把关：错误立刻抛出，不会退化成一条状态为 失败 的静默任务
    校验录制参数(算子名, 张量列表, sizeof(基础算子常量));
    // 张量句柄按值进闭包：shared_ptr 让底层缓冲与描述符集活到任务结束，
    // 即使 vk张量 本身先析构也不会让在途录制踩空。
    // 命令流同样不用调用方给：工作线程把自己那条命令流的命令缓冲交给回调。
    return 提交(流水线阶段::计算,
                [算子名, 张量列表 = std::move(张量列表)](VkCommandBuffer 命令缓冲) {
                    获取().录制到命令流(命令缓冲, 算子名, 张量列表, 基础算子常量{});
                },
                依赖, 优先级);
}

inline std::uint64_t 流水线调度器::提交自定义算子(
    const std::string& 算子名, std::vector<张量句柄> 张量列表,
    const void* 推送数据, std::uint32_t 推送字节,
    派发规格 派发,
    const std::vector<std::uint64_t>& 依赖, int 优先级)
{
    校验录制参数(算子名, 张量列表, 推送字节);
    // 推送数据需要拷贝进闭包，因为调用方的栈帧可能在任务执行前就销毁了
    std::vector<std::uint8_t> 推送副本(推送字节);
    if (推送字节 > 0 && 推送数据 != nullptr) {
        std::memcpy(推送副本.data(), 推送数据, 推送字节);
    }
    return 提交(流水线阶段::计算,
                [算子名, 张量列表 = std::move(张量列表),
                 推送副本 = std::move(推送副本), 推送字节, 派发](VkCommandBuffer 命令缓冲) {
                    获取().录制自定义派发(命令缓冲, 算子名, 张量列表,
                                          推送副本.data(), 推送字节, 派发);
                },
                依赖, 优先级);
}

// ---------------------------------------------------------------------------
// 多核 pass：一次提交录进多个 dispatch
// ---------------------------------------------------------------------------
inline void 流水线调度器::校验算子串(const std::vector<算子步骤>& 步骤列表) const {
    if (步骤列表.empty()) throw std::runtime_error("算子串为空，没有要录制的步骤");
    for (std::size_t i = 0; i < 步骤列表.size(); ++i) {
        const 算子步骤& 步骤 = 步骤列表[i];
        try {
            校验录制参数(步骤.算子名, 步骤.张量列表,
                         static_cast<std::uint32_t>(步骤.推送常量.size()));
        } catch (const std::exception& 异常) {
            throw std::runtime_error("算子串第 " + std::to_string(i) + " 步（"
                                     + 步骤.算子名 + "）：" + 异常.what());
        }
    }
}

inline void 流水线调度器::录制算子串到命令流(
    VkCommandBuffer 命令缓冲, const std::vector<算子步骤>& 步骤列表)
{
    // 算子串是独立命令缓冲、独立提交：前一次提交的写对本次首步不保证可见
    // （实际在 780M 上 T=128 会偶发读到旧值），所以串首要补一道 compute 屏障。
    {
        VkMemoryBarrier 屏障{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        屏障.srcAccessMask  = VK_ACCESS_SHADER_WRITE_BIT;
        屏障.dstAccessMask  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(命令缓冲, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1u, &屏障, 0u, nullptr, 0u, nullptr);
    }
    for (std::size_t i = 0; i < 步骤列表.size(); ++i) {
        const 算子步骤& 步骤 = 步骤列表[i];
        if (i != 0 && !步骤.不需前置屏障) {
            VkMemoryBarrier 屏障{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            屏障.srcAccessMask  = VK_ACCESS_SHADER_WRITE_BIT;
            屏障.dstAccessMask  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(命令缓冲, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1u, &屏障, 0u, nullptr, 0u, nullptr);
        }
        录制自定义派发(命令缓冲, 步骤.算子名, 步骤.张量列表,
                       步骤.推送常量.data(),
                       static_cast<std::uint32_t>(步骤.推送常量.size()),
                       步骤.派发);
    }
}

inline void 流水线调度器::录制算子串(const std::vector<算子步骤>& 步骤列表) {
    校验算子串(步骤列表);
    std::lock_guard<std::mutex> 守卫(同步命令流锁_);
    取得同步命令流().执行([this, &步骤列表](VkCommandBuffer 命令缓冲) {
        录制算子串到命令流(命令缓冲, 步骤列表);
    });
}

inline std::uint64_t 流水线调度器::提交算子串(
    std::vector<算子步骤> 步骤列表,
    const std::vector<std::uint64_t>& 依赖, int 优先级)
{
    校验算子串(步骤列表);
    // 步骤（含各自的张量句柄与推送字节）整体搬进闭包：调用方的栈与向量都先销毁，
    // 句柄里的 shared_ptr 保证缓冲活到录制结束。
    return 提交(流水线阶段::计算,
                [步骤列表 = std::move(步骤列表)](VkCommandBuffer 命令缓冲) {
                    获取().录制算子串到命令流(命令缓冲, 步骤列表);
                },
                依赖, 优先级);
}

} // namespace vk张量系统
