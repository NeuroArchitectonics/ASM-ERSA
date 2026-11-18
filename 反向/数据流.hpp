#pragma once
// 异步流式 SFT 数据管线（Belle instruction/input/output/history 格式）
//
// 线程模型：
//   读线程：逐行读 JSONL → 解析 → 有界原始队列
//   分词线程 x N：从原始队列取 → ChatML 拼接 → 分词 → prompt 掩码 → 有界样本队列
//   训练线程：取样本/取批次，队列满时自动反压，不会把 17 GB 数据读进内存
//
// 标签约定：prompt（system/history/当前 user/assistant 头）全部 -100，
//           只监督 assistant 的 output 与结尾 <|im_end|>。
// 分词器：用 C:\分词器 的 C ABI DLL（tok_create/tok_encode/tok_free），
//         每个分词线程持有一个独立句柄；tok_create 会临时改 CWD，故在启动线程前串行创建。
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sft数据 {

// ─────────────────────────────────────────────────────────────
// UTF-8 / 宽字符
// ─────────────────────────────────────────────────────────────
inline std::wstring Utf8转宽(const std::string& 文本) {
#ifdef _WIN32
    if (文本.empty()) return {};
    const int 长度 = MultiByteToWideChar(CP_UTF8, 0, 文本.data(),
                                        static_cast<int>(文本.size()), nullptr, 0);
    if (长度 <= 0) throw std::runtime_error("UTF-8 转宽失败");
    std::wstring 结果(static_cast<std::size_t>(长度), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, 文本.data(), static_cast<int>(文本.size()),
                        结果.data(), 长度);
    return 结果;
#else
    throw std::runtime_error("当前实现只支持 Windows 分词器 DLL");
#endif
}

inline std::filesystem::path Utf8路径(const std::string& 文本) {
#ifdef _WIN32
    return std::filesystem::path(Utf8转宽(文本));
#else
    return std::filesystem::path(文本);
#endif
}

// ─────────────────────────────────────────────────────────────
// 最小 JSON 解析器（只要求正确，不追求极限性能）
// ─────────────────────────────────────────────────────────────
struct Json值 {
    enum class 类型 { 空, 布尔, 数字, 字符串, 数组, 对象 };
    类型 type = 类型::空;
    bool 布尔 = false;
    double 数字 = 0.0;
    std::string 字符串;
    std::vector<Json值> 数组;
    std::vector<std::pair<std::string, Json值>> 对象;

    const Json值* 取(const std::string& 键) const {
        if (type != 类型::对象) return nullptr;
        for (const auto& [k, v] : 对象)
            if (k == 键) return &v;
        return nullptr;
    }
    const std::string* 取字符串(const std::string& 键) const {
        const Json值* v = 取(键);
        return (v && v->type == 类型::字符串) ? &v->字符串 : nullptr;
    }
};

class Json解析器 {
public:
    explicit Json解析器(std::string_view 文本)
        : 当前_(文本.data()), 结束_(文本.data() + 文本.size()) {}

    bool 解析(Json值& 值) {
        跳过空白();
        if (!解析值(值, 0)) return false;
        跳过空白();
        return 当前_ == 结束_;
    }

private:
    const char* 当前_;
    const char* 结束_;

    void 跳过空白() {
        while (当前_ < 结束_ && (*当前_ == ' ' || *当前_ == '\t' ||
                                 *当前_ == '\n' || *当前_ == '\r'))
            ++当前_;
    }

    bool 解析值(Json值& 值, int 深度) {
        if (深度 > 64 || 当前_ >= 结束_) return false;
        跳过空白();
        switch (*当前_) {
        case '{': return 解析对象(值, 深度 + 1);
        case '[': return 解析数组(值, 深度 + 1);
        case '"': 值.type = Json值::类型::字符串; return 解析字符串(值.字符串);
        case 't':
            if (结束_ - 当前_ >= 4 && std::string_view(当前_, 4) == "true") {
                当前_ += 4; 值.type = Json值::类型::布尔; 值.布尔 = true; return true;
            }
            return false;
        case 'f':
            if (结束_ - 当前_ >= 5 && std::string_view(当前_, 5) == "false") {
                当前_ += 5; 值.type = Json值::类型::布尔; 值.布尔 = false; return true;
            }
            return false;
        case 'n':
            if (结束_ - 当前_ >= 4 && std::string_view(当前_, 4) == "null") {
                当前_ += 4; 值.type = Json值::类型::空; return true;
            }
            return false;
        default:
            return 解析数字(值);
        }
    }

    bool 解析对象(Json值& 值, int 深度) {
        ++当前_;   // '{'
        值.type = Json值::类型::对象;
        跳过空白();
        if (当前_ < 结束_ && *当前_ == '}') { ++当前_; return true; }
        while (当前_ < 结束_) {
            std::string 键;
            if (!解析字符串(键)) return false;
            跳过空白();
            if (当前_ >= 结束_ || *当前_ != ':') return false;
            ++当前_;
            Json值 子值;
            if (!解析值(子值, 深度)) return false;
            值.对象.emplace_back(std::move(键), std::move(子值));
            跳过空白();
            if (当前_ >= 结束_) return false;
            if (*当前_ == ',') { ++当前_; 跳过空白(); continue; }
            if (*当前_ == '}') { ++当前_; return true; }
            return false;
        }
        return false;
    }

    bool 解析数组(Json值& 值, int 深度) {
        ++当前_;   // '['
        值.type = Json值::类型::数组;
        跳过空白();
        if (当前_ < 结束_ && *当前_ == ']') { ++当前_; return true; }
        while (当前_ < 结束_) {
            Json值 子值;
            if (!解析值(子值, 深度)) return false;
            值.数组.push_back(std::move(子值));
            跳过空白();
            if (当前_ >= 结束_) return false;
            if (*当前_ == ',') { ++当前_; 跳过空白(); continue; }
            if (*当前_ == ']') { ++当前_; return true; }
            return false;
        }
        return false;
    }

    bool 解析字符串(std::string& 输出) {
        if (当前_ >= 结束_ || *当前_ != '"') return false;
        ++当前_;
        while (当前_ < 结束_) {
            unsigned char c = static_cast<unsigned char>(*当前_++);
            if (c == '"') return true;
            if (c == '\\') {
                if (当前_ >= 结束_) return false;
                switch (*当前_++) {
                case '"': 输出.push_back('"'); break;
                case '\\': 输出.push_back('\\'); break;
                case '/': 输出.push_back('/'); break;
                case 'b': 输出.push_back('\b'); break;
                case 'f': 输出.push_back('\f'); break;
                case 'n': 输出.push_back('\n'); break;
                case 'r': 输出.push_back('\r'); break;
                case 't': 输出.push_back('\t'); break;
                case 'u': {
                    std::uint32_t 码点 = 0;
                    if (!解析4位十六进制(码点)) return false;
                    if (码点 >= 0xD800 && 码点 <= 0xDBFF &&
                        结束_ - 当前_ >= 6 && 当前_[0] == '\\' && 当前_[1] == 'u') {
                        当前_ += 2;
                        std::uint32_t 低位 = 0;
                        if (!解析4位十六进制(低位)) return false;
                        if (低位 >= 0xDC00 && 低位 <= 0xDFFF)
                            码点 = 0x10000 + ((码点 - 0xD800) << 10) + (低位 - 0xDC00);
                        else
                            码点 = 0xFFFD;
                    } else if (码点 >= 0xD800 && 码点 <= 0xDFFF) {
                        码点 = 0xFFFD;
                    }
                    追加UTF8(码点, 输出);
                    break;
                }
                default: return false;
                }
            } else if (c < 0x20) {
                return false;
            } else {
                输出.push_back(static_cast<char>(c));
            }
        }
        return false;
    }

    bool 解析4位十六进制(std::uint32_t& 值) {
        if (结束_ - 当前_ < 4) return false;
        值 = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = *当前_++;
            std::uint32_t d;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
            else return false;
            值 = (值 << 4) | d;
        }
        return true;
    }

    static void 追加UTF8(std::uint32_t 码点, std::string& 输出) {
        if (码点 <= 0x7F) {
            输出.push_back(static_cast<char>(码点));
        } else if (码点 <= 0x7FF) {
            输出.push_back(static_cast<char>(0xC0 | (码点 >> 6)));
            输出.push_back(static_cast<char>(0x80 | (码点 & 0x3F)));
        } else if (码点 <= 0xFFFF) {
            输出.push_back(static_cast<char>(0xE0 | (码点 >> 12)));
            输出.push_back(static_cast<char>(0x80 | ((码点 >> 6) & 0x3F)));
            输出.push_back(static_cast<char>(0x80 | (码点 & 0x3F)));
        } else {
            输出.push_back(static_cast<char>(0xF0 | (码点 >> 18)));
            输出.push_back(static_cast<char>(0x80 | ((码点 >> 12) & 0x3F)));
            输出.push_back(static_cast<char>(0x80 | ((码点 >> 6) & 0x3F)));
            输出.push_back(static_cast<char>(0x80 | (码点 & 0x3F)));
        }
    }

    bool 解析数字(Json值& 值) {
        const char* 开始 = 当前_;
        if (当前_ < 结束_ && (*当前_ == '-' || *当前_ == '+')) ++当前_;
        bool 有数字 = false;
        while (当前_ < 结束_ && *当前_ >= '0' && *当前_ <= '9') { ++当前_; 有数字 = true; }
        if (当前_ < 结束_ && *当前_ == '.') {
            ++当前_;
            while (当前_ < 结束_ && *当前_ >= '0' && *当前_ <= '9') { ++当前_; 有数字 = true; }
        }
        if (有数字 && 当前_ < 结束_ && (*当前_ == 'e' || *当前_ == 'E')) {
            ++当前_;
            if (当前_ < 结束_ && (*当前_ == '-' || *当前_ == '+')) ++当前_;
            while (当前_ < 结束_ && *当前_ >= '0' && *当前_ <= '9') ++当前_;
        }
        if (!有数字) return false;
        值.type = Json值::类型::数字;
        std::from_chars(开始, 当前_, 值.数字);
        return true;
    }
};

// ─────────────────────────────────────────────────────────────
// Belle 原始样本
// ─────────────────────────────────────────────────────────────
struct 原文样本 {
    std::string instruction;
    std::string input;
    std::string output;
    std::vector<std::pair<std::string, std::string>> history;
    std::uint64_t 行号 = 0;
};

inline bool 解析Belle行(const std::string& 行, std::uint64_t 行号, 原文样本& 出) {
    Json值 根;
    if (!Json解析器(行).解析(根) || 根.type != Json值::类型::对象) return false;
    出 = 原文样本{};
    出.行号 = 行号;
    if (const auto* v = 根.取字符串("instruction")) 出.instruction = *v;
    if (const auto* v = 根.取字符串("input")) 出.input = *v;
    if (const auto* v = 根.取字符串("output")) 出.output = *v;
    if (const Json值* h = 根.取("history"); h && h->type == Json值::类型::数组) {
        for (const Json值& 轮 : h->数组) {
            if (轮.type != Json值::类型::数组 || 轮.数组.size() < 2) continue;
            const Json值& 用户 = 轮.数组[0];
            const Json值& 助手 = 轮.数组[1];
            if (用户.type == Json值::类型::字符串 && 助手.type == Json值::类型::字符串)
                出.history.emplace_back(用户.字符串, 助手.字符串);
        }
    }
    return !出.output.empty();
}

// ─────────────────────────────────────────────────────────────
// C ABI 分词器 DLL
// ─────────────────────────────────────────────────────────────
class 分词器DLL {
public:
    using 创建函数 = void* (*)(const wchar_t*);
    using 词表大小函数 = int (*)(void*);
    using 编码函数 = int (*)(void*, const char*, int*, int);
    using 释放函数 = void (*)(void*);

    分词器DLL() = default;
    分词器DLL(const 分词器DLL&) = delete;
    分词器DLL& operator=(const 分词器DLL&) = delete;
    ~分词器DLL() {
#ifdef _WIN32
        if (模块_) FreeLibrary(模块_);
#endif
    }

    void 加载(const std::string& dll路径) {
#ifdef _WIN32
        const std::wstring 宽路径 = Utf8转宽(dll路径);
        模块_ = LoadLibraryW(宽路径.c_str());
        if (!模块_) throw std::runtime_error("无法加载分词器 DLL: " + dll路径);
        创建_ = reinterpret_cast<创建函数>(GetProcAddress(模块_, "tok_create"));
        词表大小_ = reinterpret_cast<词表大小函数>(GetProcAddress(模块_, "tok_vocab_size"));
        编码_ = reinterpret_cast<编码函数>(GetProcAddress(模块_, "tok_encode"));
        释放_ = reinterpret_cast<释放函数>(GetProcAddress(模块_, "tok_free"));
        if (!创建_ || !词表大小_ || !编码_ || !释放_)
            throw std::runtime_error("分词器 DLL 缺少 tok_create/tok_vocab_size/tok_encode/tok_free");
#else
        throw std::runtime_error("当前实现只支持 Windows 分词器 DLL");
#endif
    }

    void* 新建(const std::string& 模型目录) const {
        const std::wstring 宽目录 = Utf8转宽(模型目录);
        void* 句柄 = 创建_(宽目录.c_str());
        if (!句柄) throw std::runtime_error("分词器初始化失败: " + 模型目录);
        return 句柄;
    }

    int 词表大小(void* 句柄) const { return 词表大小_(句柄); }

    std::vector<int> 编码(void* 句柄, const std::string& 文本) const {
        const int 长度 = 编码_(句柄, 文本.c_str(), nullptr, 0);
        if (长度 <= 0) return {};
        std::vector<int> 标识(static_cast<std::size_t>(长度));
        const int 实际 = 编码_(句柄, 文本.c_str(), 标识.data(), 长度);
        if (实际 <= 0) return {};
        标识.resize(std::min(长度, 实际));
        return 标识;
    }

    void 释放(void* 句柄) const { if (句柄) 释放_(句柄); }

private:
#ifdef _WIN32
    HMODULE 模块_ = nullptr;
#endif
    创建函数 创建_ = nullptr;
    词表大小函数 词表大小_ = nullptr;
    编码函数 编码_ = nullptr;
    释放函数 释放_ = nullptr;
};

// ─────────────────────────────────────────────────────────────
// 对外数据类型
// ─────────────────────────────────────────────────────────────
struct 样本 {
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> labels;      // prompt / padding 为 -100
    std::string 用户;
    std::string 回答;
    std::uint64_t 行号 = 0;
};

struct 批次 {
    std::uint32_t 批大小 = 0;
    std::uint32_t 序列长度 = 0;
    std::vector<std::int32_t> input_ids;   // [B*T]
    std::vector<std::int32_t> labels;      // [B*T]
    std::vector<std::uint8_t> attention;   // [B*T]，1=真实 token
    std::uint32_t 有效token数 = 0;
};

struct 配置 {
    std::string 数据路径;
    std::string 分词器目录;                 // 含 tokenizer.json
    std::string 分词器DLL;                  // 空则读 ASM_ERSA_TOKENIZER_DLL，再退回 .cache/tok_dll.dll
    std::uint32_t 最大序列长度 = 2048;
    std::uint32_t 批大小 = 4;
    std::uint32_t 原始队列容量 = 1024;
    std::uint32_t 样本队列容量 = 256;
    std::uint32_t 分词线程数 = 2;
    std::uint64_t 最大样本数 = 0;           // 0 = 读到文件尾
    bool 加系统提示 = true;
    std::string 系统提示 = "You are a helpful assistant.";
};

struct 统计 {
    std::uint64_t 已读行 = 0;
    std::uint64_t 已解析 = 0;
    std::uint64_t 已分词 = 0;
    std::uint64_t 跳过 = 0;
    std::uint64_t 总token = 0;
    std::size_t 原始队列 = 0;
    std::size_t 样本队列 = 0;
    double 已用秒 = 0.0;
};

// ─────────────────────────────────────────────────────────────
// 数据流
// ─────────────────────────────────────────────────────────────
class Sft数据流 {
public:
    explicit Sft数据流(配置 参数) : 配置_(std::move(参数)) {
        if (配置_.数据路径.empty()) throw std::invalid_argument("数据路径为空");
        if (配置_.分词器目录.empty()) {
            if (const char* 环境 = std::getenv("ASM_ERSA_TOKENIZER")) 配置_.分词器目录 = 环境;
        }
        if (配置_.分词器目录.empty()) throw std::invalid_argument("分词器目录为空");
        if (配置_.最大序列长度 < 16) throw std::invalid_argument("最大序列长度至少 16");
        配置_.分词线程数 = std::clamp<std::uint32_t>(配置_.分词线程数, 1u, 8u);
        配置_.原始队列容量 = std::max<std::uint32_t>(配置_.原始队列容量, 1u);
        配置_.样本队列容量 = std::max<std::uint32_t>(配置_.样本队列容量, 1u);

        // 先确认文件能打开，失败在构造期暴露
        {
            std::ifstream 检查(Utf8路径(配置_.数据路径), std::ios::binary);
            if (!检查) throw std::runtime_error("无法打开数据文件: " + 配置_.数据路径);
        }

        std::string dll路径 = 配置_.分词器DLL;
        if (dll路径.empty()) {
            if (const char* 环境 = std::getenv("ASM_ERSA_TOKENIZER_DLL")) dll路径 = 环境;
        }
        if (dll路径.empty()) dll路径 = ".cache/tok_dll.dll";
        dll_.加载(dll路径);

        try {
            for (std::uint32_t i = 0; i < 配置_.分词线程数; ++i)
                分词器_.push_back(dll_.新建(配置_.分词器目录));
            im_end_ = 查单token("<|im_end|>");
            pad_id_ = 查单token("<|endoftext|>");
            词表大小_ = dll_.词表大小(分词器_.front());
            if (词表大小_ <= 0) throw std::runtime_error("分词器词表大小无效");

            开始时间_ = std::chrono::steady_clock::now();
            活跃分词线程_.store(配置_.分词线程数);
            读取线程_ = std::thread([this] { 读取循环(); });
            for (std::uint32_t i = 0; i < 配置_.分词线程数; ++i)
                工作线程_.emplace_back([this, i] { 分词循环(i); });
        } catch (...) {
            清理();
            throw;
        }
    }

    ~Sft数据流() { 清理(); }

    Sft数据流(const Sft数据流&) = delete;
    Sft数据流& operator=(const Sft数据流&) = delete;

    bool 取样本(样本& 出) {
        std::unique_lock 锁(样本锁_);
        样本条件_.wait(锁, [this] {
            return 停止_.load() || !样本队列_.empty() ||
                   (读取完毕_.load() && 活跃分词线程_.load() == 0);
        });
        if (样本队列_.empty()) return false;
        出 = std::move(样本队列_.front());
        样本队列_.pop_front();
        锁.unlock();
        样本条件_.notify_all();
        return true;
    }

    // 阻塞取第一个，其后只取队列里已有的；序列长度 T 不足以容纳样本时保留尾部（保答案）。
    bool 取批次(批次& 出, std::uint32_t 批大小 = 0, std::uint32_t 序列长度 = 0) {
        if (批大小 == 0) 批大小 = 配置_.批大小;
        if (序列长度 == 0) 序列长度 = 配置_.最大序列长度;
        std::vector<样本> 组;
        {
            样本 首个;
            if (!取样本(首个)) return false;
            组.push_back(std::move(首个));
        }
        while (组.size() < 批大小) {
            样本 下一个;
            {
                std::lock_guard 锁(样本锁_);
                if (样本队列_.empty()) break;
                下一个 = std::move(样本队列_.front());
                样本队列_.pop_front();
            }
            组.push_back(std::move(下一个));
        }

        出 = 批次{};
        出.批大小 = static_cast<std::uint32_t>(组.size());
        出.序列长度 = 序列长度;
        const std::size_t 总数 = static_cast<std::size_t>(出.批大小) * 序列长度;
        出.input_ids.assign(总数, pad_id_);
        出.labels.assign(总数, -100);
        出.attention.assign(总数, 0);
        for (std::uint32_t b = 0; b < 出.批大小; ++b) {
            const 样本& s = 组[b];
            const std::size_t 长度 = s.input_ids.size();
            const std::size_t 保留 = std::min<std::size_t>(长度, 序列长度);
            const std::size_t 起点 = 长度 - 保留;
            for (std::size_t t = 0; t < 保留; ++t) {
                const std::size_t 目的 = static_cast<std::size_t>(b) * 序列长度 + t;
                出.input_ids[目的] = s.input_ids[起点 + t];
                出.labels[目的] = s.labels[起点 + t];
                出.attention[目的] = 1;
            }
            出.有效token数 += static_cast<std::uint32_t>(保留);
        }
        return true;
    }

    统计 取统计() const {
        统计 s;
        s.已读行 = 已读行_.load();
        s.已解析 = 已解析_.load();
        s.已分词 = 已分词_.load();
        s.跳过 = 跳过_.load();
        s.总token = 总token_.load();
        {
            std::lock_guard 锁(原始锁_);
            s.原始队列 = 原始队列_.size();
        }
        {
            std::lock_guard 锁(样本锁_);
            s.样本队列 = 样本队列_.size();
        }
        s.已用秒 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - 开始时间_).count();
        return s;
    }

    std::int32_t 填充id() const { return pad_id_; }
    std::int32_t 结束id() const { return im_end_; }
    int 词表大小() const { return 词表大小_; }

private:
    std::int32_t 查单token(const std::string& 文本) const {
        const std::vector<int> 标识 = dll_.编码(分词器_.front(), 文本);
        if (标识.size() != 1)
            throw std::runtime_error("分词器没有把 " + 文本 + " 编码成单个特殊 token");
        return static_cast<std::int32_t>(标识[0]);
    }

    void 读取循环() {
        std::ifstream 文件(Utf8路径(配置_.数据路径), std::ios::binary);
        if (!文件) {
            跳过_.fetch_add(1);
            读取完毕_.store(true);
            原始条件_.notify_all();
            样本条件_.notify_all();
            return;
        }

        std::string 行;
        std::uint64_t 行号 = 0;
        while (!停止_.load()) {
            if (配置_.最大样本数 != 0 && 已解析_.load() >= 配置_.最大样本数) break;
            if (!std::getline(文件, 行)) break;
            ++行号;
            已读行_.fetch_add(1);
            if (!行.empty() && 行.back() == '\r') 行.pop_back();
            if (行.empty()) continue;

            原文样本 原文;
            if (!解析Belle行(行, 行号, 原文)) {
                跳过_.fetch_add(1);
                continue;
            }
            已解析_.fetch_add(1);
            if (!推入原始(std::move(原文))) break;
        }

        读取完毕_.store(true);
        原始条件_.notify_all();
        样本条件_.notify_all();
    }

    bool 推入原始(原文样本&& 原文) {
        std::unique_lock 锁(原始锁_);
        原始条件_.wait(锁, [this] {
            return 停止_.load() || 原始队列_.size() < 配置_.原始队列容量;
        });
        if (停止_.load()) return false;
        原始队列_.push_back(std::move(原文));
        锁.unlock();
        原始条件_.notify_all();
        return true;
    }

    void 分词循环(std::uint32_t 序号) {
        while (true) {
            原文样本 原文;
            {
                std::unique_lock 锁(原始锁_);
                原始条件_.wait(锁, [this] {
                    return 停止_.load() || 读取完毕_.load() || !原始队列_.empty();
                });
                if (原始队列_.empty()) {
                    if (停止_.load() || 读取完毕_.load()) break;
                    continue;
                }
                原文 = std::move(原始队列_.front());
                原始队列_.pop_front();
            }
            原始条件_.notify_all();

            样本 s;
            if (!构造样本(分词器_[序号], 原文, s)) {
                跳过_.fetch_add(1);
                continue;
            }
            if (!推入样本(std::move(s))) break;
        }

        活跃分词线程_.fetch_sub(1);
        样本条件_.notify_all();
    }

    bool 推入样本(样本&& s) {
        std::unique_lock 锁(样本锁_);
        样本条件_.wait(锁, [this] {
            return 停止_.load() || 样本队列_.size() < 配置_.样本队列容量;
        });
        if (停止_.load()) return false;
        样本队列_.push_back(std::move(s));
        锁.unlock();
        样本条件_.notify_all();
        return true;
    }

    bool 构造样本(void* 分词器, const 原文样本& 原文, 样本& 出) {
        std::string 用户 = 原文.instruction;
        if (!原文.input.empty()) {
            if (!用户.empty()) 用户 += '\n';
            用户 += 原文.input;
        }
        if (用户.empty()) return false;

        std::string 提示;
        if (配置_.加系统提示) {
            提示 += "<|im_start|>system\n";
            提示 += 配置_.系统提示;
            提示 += "<|im_end|>\n";
        }
        for (const auto& [轮用户, 轮助手] : 原文.history) {
            提示 += "<|im_start|>user\n";
            提示 += 轮用户;
            提示 += "<|im_end|>\n<|im_start|>assistant\n";
            提示 += 轮助手;
            提示 += "<|im_end|>\n";
        }
        提示 += "<|im_start|>user\n";
        提示 += 用户;
        提示 += "<|im_end|>\n<|im_start|>assistant\n";

        std::string 回答文本 = 原文.output;
        回答文本 += "<|im_end|>";

        std::vector<int> 前缀 = dll_.编码(分词器, 提示);
        std::vector<int> 回答 = dll_.编码(分词器, 回答文本);
        if (回答.empty()) return false;

        const std::size_t 上限 = 配置_.最大序列长度;
        const std::size_t 最小前缀 = 8;
        // 长回答优先截断，至少保住 assistant 头的 8 个 prompt token；
        // 否则长输出会把前缀挤到 0，整条样本被丢弃。
        if (回答.size() > 上限 - 最小前缀) {
            回答.resize(上限 - 最小前缀 - 1);
            回答.push_back(im_end_);
        }
        const std::size_t 可留前缀 = 上限 - 回答.size();
        if (可留前缀 < 最小前缀) return false;
        if (前缀.size() > 可留前缀)
            前缀.erase(前缀.begin(), 前缀.begin() + (前缀.size() - 可留前缀));

        出 = 样本{};
        出.行号 = 原文.行号;
        出.用户 = std::move(用户);
        出.回答 = 原文.output;
        出.input_ids.reserve(前缀.size() + 回答.size());
        出.labels.reserve(前缀.size() + 回答.size());
        for (int id : 前缀) {
            出.input_ids.push_back(static_cast<std::int32_t>(id));
            出.labels.push_back(-100);
        }
        for (int id : 回答) {
            出.input_ids.push_back(static_cast<std::int32_t>(id));
            出.labels.push_back(static_cast<std::int32_t>(id));
        }
        已分词_.fetch_add(1);
        总token_.fetch_add(出.input_ids.size());
        return true;
    }

    void 清理() {
        停止_.store(true);
        原始条件_.notify_all();
        样本条件_.notify_all();
        if (读取线程_.joinable()) 读取线程_.join();
        for (auto& t : 工作线程_) if (t.joinable()) t.join();
        工作线程_.clear();
        for (void* 句柄 : 分词器_) dll_.释放(句柄);
        分词器_.clear();
    }

    配置 配置_;
    分词器DLL dll_;
    std::vector<void*> 分词器_;
    std::int32_t im_end_ = 0;
    std::int32_t pad_id_ = 0;
    int 词表大小_ = 0;

    std::thread 读取线程_;
    std::vector<std::thread> 工作线程_;

    mutable std::mutex 原始锁_;
    std::condition_variable 原始条件_;
    std::deque<原文样本> 原始队列_;

    mutable std::mutex 样本锁_;
    std::condition_variable 样本条件_;
    std::deque<样本> 样本队列_;

    std::atomic<bool> 停止_{false};
    std::atomic<bool> 读取完毕_{false};
    std::atomic<std::uint32_t> 活跃分词线程_{0};
    std::atomic<std::uint64_t> 已读行_{0};
    std::atomic<std::uint64_t> 已解析_{0};
    std::atomic<std::uint64_t> 已分词_{0};
    std::atomic<std::uint64_t> 跳过_{0};
    std::atomic<std::uint64_t> 总token_{0};
    std::chrono::steady_clock::time_point 开始时间_ = std::chrono::steady_clock::now();
};

} // namespace sft数据
