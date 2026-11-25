// ============================================================================
//  着色编译器.hpp
//  Vulkan 底层封装 · 指定源目录 → 指定产物目录（.spv + .spv.hpp）
//
//  设计要点：
//    1. 只管编译期。输入一批 .glsl/.comp 源，输出 SPIR-V 二进制 .spv
//       和同名 .spv.hpp 元信息头，两者必须同源，头里带 SPIR-V 指纹。
//    2. 编译经由外部 glslangValidator / glslc，本文件不链 glslang。
//       想用库内编译只改 调用编译器() 一处。
//    3. 报错规范化成 "路径:行:列: error: 说明"，IDE 构建面板可直接点击跳转。
//    4. 编译失败也要写出 .spv.hpp，内容是一行 #error —— 否则残留的旧产物
//       会被继续编进宿主，表现为结果错但没有报错。
//    5. 元信息从 GLSL 源解析（含本地 #include 展开），因此强制要求
//       所有 buffer/uniform 块显式写 set 与 binding，缺失即视为工具错误。
//    6. 单文件头，inline 实现，中文标识符与注释。
// ============================================================================
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define VKTS_POPEN _popen
#define VKTS_PCLOSE _pclose
#else
#define VKTS_POPEN popen
#define VKTS_PCLOSE pclose
#endif

namespace vk张量系统 {

// ---------------------------------------------------------------------------
// 单个算子的解析结果
// ---------------------------------------------------------------------------
struct 算子定义 {
    std::string   算子名;              // 文件名去扩展名，英文
    std::string   源路径;
    std::string   产物路径;            // <产物目录>/<算子名>.spv
    std::string   头路径;              // <产物目录>/<算子名>.spv.hpp
    std::uint32_t 工作组大小 = 0;
    std::uint32_t 绑定数     = 0;
    std::uint32_t 推送常量字节 = 0;
    std::uint32_t 产物指纹   = 0;
    std::vector<std::string> 绑定清单;  // "槽位0:BufA" 形式，供诊断
    std::vector<std::string> 包含文件;
    bool          成功 = false;
    bool          跳过 = false;      // 公共源（无入口函数），不参与编译
    std::string   诊断;
};

// ---------------------------------------------------------------------------
// 编译选项
// ---------------------------------------------------------------------------
struct 编译选项 {
    std::string 源目录;                       // 必填：*.glsl / *.comp 源
    std::string 产物目录;                     // 必填：.spv 与 .spv.hpp 落这里
    std::string 编译器路径 = "glslangValidator";
    std::string 优化工具路径 = "spirv-opt";   // 空字符串则跳过优化
    std::string 目标环境 = "vulkan1.2";
    bool        递归 = true;
    bool        失败即抛出 = true;
    bool        允许源已是SPIRV = true;       // 源文件已是二进制则只做解析与拷贝
};

// ---------------------------------------------------------------------------
// 着色编译器
// ---------------------------------------------------------------------------
class 着色编译器 {
public:
    // 主入口：源目录 → 产物目录。返回每个算子的处理结果
    std::vector<算子定义> 编译全部(编译选项 选项);

    // 单独暴露，便于构建脚本先做语法检查再决定是否编译
    static bool 解析源文件(const std::string& 路径, 算子定义& 输出, std::string& 原因);

    // 已存在且带 SPIR-V 魔数的文件视为二进制产物，不再当 GLSL 源解析
    static bool 是spirv二进制(const std::string& 路径);

    // 没有 void main 的文件是公共源（例如 common.glsl），只当被包含，不出产物
    static bool 含入口函数(const std::string& 路径);

    std::string const& 最后诊断() const { return 诊断汇总_; }
    std::size_t        成功数() const { return 成功数_; }

private:
    // 唯一的外部编译器调用点
    static bool 调用编译器(const 编译选项& 选项, const std::string& 源,
                           const std::string& 输出, std::string& 日志);

    static std::string 规范化文本(std::string 文本);          // 去注释但保留行号
    static std::string 展开包含(const std::string& 路径, const std::string& 原文,
                                算子定义& 定义);
    static bool 解析布局(const std::string& 原文, 算子定义& 定义, std::string& 原因);
    static std::uint32_t 计算指纹(const std::vector<std::uint32_t>& 字);
    static void 写元信息头(const 算子定义& 定义);
    static void 写失败头(const 算子定义& 定义);

    std::string 诊断汇总_;
    std::size_t 成功数_ = 0;
};

// ============================================================================
// 实现
// ============================================================================

inline bool 着色编译器::是spirv二进制(const std::string& 路径) {
    std::ifstream 流(路径, std::ios::binary);
    if (!流) return false;
    unsigned char 头[4] = {0, 0, 0, 0};
    流.read(reinterpret_cast<char*>(头), 4);
    if (流.gcount() < 4) return false;
    // 小端 0x07 0x23 0x02 0x03
    return 头[0] == 0x03 && 头[1] == 0x02 && 头[2] == 0x23 && 头[3] == 0x07;
}

inline bool 着色编译器::含入口函数(const std::string& 路径) {
    std::ifstream 流(路径, std::ios::binary);
    if (!流) return false;
    std::ostringstream 汇;
    汇 << 流.rdbuf();
    static const std::regex 入口(R"(void\s+main\s*\()");
    return std::regex_search(汇.str(), 入口);
}

inline std::string 着色编译器::规范化文本(std::string 文本) {
    // 注释替换为等长空格（换行保留），行号与列号才对得上原文件
    for (std::size_t i = 0; i < 文本.size();) {
        if (文本[i] == '/' && i + 1 < 文本.size() && 文本[i + 1] == '/') {
            while (i < 文本.size() && 文本[i] != '\n') { 文本[i] = ' '; ++i; }
        } else if (文本[i] == '/' && i + 1 < 文本.size() && 文本[i + 1] == '*') {
            std::size_t 起 = i;
            std::size_t 止 = 文本.find("*/", i + 2);
            if (止 == std::string::npos) 止 = 文本.size(); else 止 += 2;
            for (std::size_t k = 起; k < 止; ++k) if (文本[k] != '\n') 文本[k] = ' ';
            i = 止;
        } else {
            ++i;
        }
    }
    return 文本;
}

inline std::string 着色编译器::展开包含(const std::string& 路径, const std::string& 原文,
                                        算子定义& 定义) {
    static const std::regex 包含表达式(R"rx(^\s*#\s*include\s*"([^"]+))rx");
    std::string 结果;
    std::istringstream 流(原文);
    std::string 行;
    std::filesystem::path 当前(路径);

    while (std::getline(流, 行)) {
        std::smatch 匹配;
        if (std::regex_search(行, 匹配, 包含表达式)) {
            std::filesystem::path 目标 =
                std::filesystem::path(当前.parent_path()).lexically_normal() / 匹配[1].str();
            定义.包含文件.push_back(目标.string());
            if (std::filesystem::exists(目标)) {
                std::ifstream 子(目标, std::ios::binary);
                std::ostringstream 汇;
                汇 << 子.rdbuf();
                结果 += 规范化文本(汇.str());   // 嵌套包含不再递归，公共头里不该有
                结果 += "\n";
                continue;
            }
            // 缺失的公共头交给编译器报错，这里保留原行以便行号定位
        }
        结果 += 行;
        结果 += "\n";
    }
    return 结果;
}

// std430 标量/向量/矩阵的 (尺寸, 对齐)。未知类型返回 0 由调用方报错
inline void 类型尺寸(const std::string& 类型, std::uint32_t& 尺寸, std::uint32_t& 对齐) {
    尺寸 = 0; 对齐 = 0;
    static const std::regex 向量(R"(^(?:b?f?i?u?vec|dvec)([234])$)");
    static const std::regex 矩阵(R"(^(?:mat|dmat|smat)([234])x([234])$)");
    std::smatch 匹配;

    if (类型 == "float" || 类型 == "int" || 类型 == "uint" || 类型 == "bool") { 尺寸 = 4; 对齐 = 4; return; }
    if (类型 == "double" || 类型 == "int64_t" || 类型 == "uint64_t") { 尺寸 = 8; 对齐 = 8; return; }
    if (类型 == "i8" || 类型 == "u8" || 类型 == "byte") { 尺寸 = 1; 对齐 = 1; return; }
    if (类型 == "i16" || 类型 == "u16" || 类型 == "half") { 尺寸 = 2; 对齐 = 2; return; }
    if (std::regex_match(类型, 匹配, 向量)) {
        std::uint32_t 分量 = 0, 单元 = 0;
        类型尺寸(类型[0] == 'd' ? "double" : "float", 单元, 分量);
        std::uint32_t 维 = static_cast<std::uint32_t>(std::stoi(匹配[1].str()));
        尺寸 = 单元 * 维;
        对齐 = (维 == 3 ? 16u : 单元 * (维 == 2 ? 2u : 1u));
        if (类型[0] == 'd') 对齐 = 8u * (维 == 3 ? 2u : 维 == 2 ? 2u : 1u);
        return;
    }
    if (std::regex_match(类型, 匹配, 矩阵)) {
        std::uint32_t 列 = static_cast<std::uint32_t>(std::stoi(匹配[1].str()));
        std::uint32_t 行 = static_cast<std::uint32_t>(std::stoi(匹配[2].str()));
        std::uint32_t 单元 = 0, 分量 = 0;
        类型尺寸(类型[0] == 'd' ? "double" : "float", 单元, 分量);
        std::uint32_t 列长 = 行 * 单元;
        std::uint32_t 列对齐 = 2u * 单元;
        std::uint32_t 跨距 = (列长 % 列对齐) ? (列长 + 列对齐 - 列长 % 列对齐) : 列长;
        尺寸 = 跨距 * 列;
        对齐 = 列对齐;
        return;
    }
}

inline bool 着色编译器::解析布局(const std::string& 原文, 算子定义& 定义, std::string& 原因) {
    static const std::regex 布局串(R"(layout\s*\(([^)]*)\))");
    std::sregex_iterator 它(原文.begin(), 原文.end(), 布局串), 尾;

    // local_size：优先取显式常量值，_id 只用来确认支持运行期覆盖
    定义.工作组大小 = 0;
    for (std::sregex_iterator 项 = 它; 项 != 尾; ++项) {
        std::string 参数 = (*项)[1].str();
        std::regex 键值(R"((\w+)\s*=\s*(\d+))");
        for (std::sregex_iterator k(参数.begin(), 参数.end(), 键值), ke; k != ke; ++k) {
            std::string 名 = (*k)[1].str();
            if (名 == "local_size_x") 定义.工作组大小 = std::stoul((*k)[2].str());
            if (名 == "local_size_y" || 名 == "local_size_z") {
                std::uint32_t 其他 = std::stoul((*k)[2].str());
                if (其他 != 1u && 定义.工作组大小 == 0) 定义.工作组大小 = 其他;
            }
        }
    }
    if (定义.工作组大小 == 0) { 原因 = "源文件里没有 layout(local_size_x = N) in;"; return false; }
    if (定义.工作组大小 % 2u != 0 && 定义.工作组大小 != 1u) {
        std::fprintf(stderr, "[着色编译器] %s：工作组 %u 为奇数，向量化访存通常吃不满\n",
                     定义.算子名.c_str(), 定义.工作组大小);
    }

    // 块声明：layout(...) buffer/uniform 名字 { ... } 实例名;
    // readonly/writeonly/coherent/volatile/nonprivate 等限定符夹在 layout 与存储类之间，必须容忍
    static const std::regex 块(R"rx(layout\s*\(([^)]*)\)\s*(?:readonly\s+|writeonly\s+|coherent\s+|volatile\s+|nonprivate\s+)*(buffer|uniform)\s*(\w*)\s*\{([^}]*)\}\s*(\w*)\s*;)rx");
    定义.绑定数 = 0;
    定义.推送常量字节 = 0;
    定义.绑定清单.clear();

    for (std::sregex_iterator b(原文.begin(), 原文.end(), 块), be; b != be; ++b) {
        std::string 参数 = (*b)[1].str();
        std::string 块名 = (*b)[3].str();
        std::string 本体 = (*b)[4].str();
        std::string 实例 = (*b)[5].str();

        bool 是推送 = 参数.find("push_constant") != std::string::npos;
        std::smatch 槽, 集;
        std::regex 槽式(R"(binding\s*=\s*(\d+))"), 集式(R"(set\s*=\s*(\d+))");
        bool 有槽 = std::regex_search(参数, 槽, 槽式);
        bool 有集 = std::regex_search(参数, 集, 集式);

        if (是推送) {
            // 结构体尺寸按 std430 累加，末尾对齐到最大成员对齐
            std::uint32_t 偏移 = 0, 最大对齐 = 4;
            static const std::regex 字段(R"(([\w]+)\s+(\w+)(\s*\[[^\]]*\])?\s*;)");
            for (std::sregex_iterator f(本体.begin(), 本体.end(), 字段), fe; f != fe; ++f) {
                std::uint32_t 尺寸 = 0, 对齐 = 0;
                类型尺寸((*f)[1].str(), 尺寸, 对齐);
                if (尺寸 == 0) {
                    原因 = "推送常量字段 " + (*f)[2].str() + " 的类型 " + (*f)[1].str() +
                           " 无法定尺寸，请补进 类型尺寸() 或改用 uint/float";
                    return false;
                }
                if ((*f)[3].matched) {   // 数组成员：按 尺寸×长度 展开，对齐规则同标量
                    std::string 括号 = (*f)[3].str();
                    std::smatch 数;
                    std::uint32_t 长度 = 1;
                    if (std::regex_search(括号, 数, std::regex(R"rx(\[\s*(\d+)\s*\])rx")))
                        长度 = static_cast<std::uint32_t>(std::stoul(数[1].str()));
                    else { 原因 = "推送常量不允许未定长数组: " + (*f)[2].str(); return false; }
                    尺寸 *= 长度;
                }
                偏移 += (偏移 % 对齐) ? (对齐 - 偏移 % 对齐) : 0u;
                偏移 += 尺寸;
                最大对齐 = std::max(最大对齐, 对齐);
            }
            偏移 += (偏移 % 最大对齐) ? (最大对齐 - 偏移 % 最大对齐) : 0u;
            定义.推送常量字节 = 偏移;
            定义.绑定清单.push_back("push_constant:" + 块名 + "=" + std::to_string(偏移) + "B");
            continue;
        }

        if (!有槽 || !有集) {
            原因 = "块 " + 块名 + "（实例 " + 实例 + "）没有显式写 set 与 binding；"
                   "按声明顺序自动编号会在重排时静默错位，本工具拒绝生成";
            return false;
        }
        // vk底层 每个描述符集只装一个 storage buffer，且固定在 binding 0；
        // 第 i 个张量走 set i，调度器录制时按下标一次绑 N 个集。
        if (槽[1].str() != "0") {
            原因 = "块 " + 块名 + "（实例 " + 实例 + "）位于 binding " + 槽[1].str() +
                   "；每个描述符集只有一个 binding 且必须是 0，第 i 个张量写 set = i，"
                   "即 layout(set=i, binding=0)";
            return false;
        }
        ++定义.绑定数;
        定义.绑定清单.push_back("set" + 集[1].str() + " binding0:" + 实例);
    }

    if (定义.绑定数 == 0) { 原因 = "没有解析到任何 buffer/uniform 绑定"; return false; }
    // set 必须从 0 连续：调度器按张量下标 0/1/2… 绑描述符集，断号会静默错位
    std::vector<std::uint32_t> 集号;
    for (const auto& 项 : 定义.绑定清单) {
        if (项.rfind("set", 0) != 0) continue;
        const std::size_t 空格 = 项.find(' ');
        if (空格 == std::string::npos || 空格 <= 3) continue;
        集号.push_back(static_cast<std::uint32_t>(std::stoul(项.substr(3, 空格 - 3))));
    }
    std::sort(集号.begin(), 集号.end());
    for (std::uint32_t i = 0; i < 集号.size(); ++i) {
        if (集号[i] != i) {
            原因 = "set 编号不连续，第 " + std::to_string(i) + " 项是 set " +
                   std::to_string(集号[i]) + "；请改为从 0 连续递增（第 i 个张量 = set i）";
            return false;
        }
    }
    return true;
}

inline bool 着色编译器::解析源文件(const std::string& 路径, 算子定义& 定义, std::string& 原因) {
    std::ifstream 流(路径, std::ios::binary);
    if (!流) { 原因 = "打开源文件失败"; return false; }
    std::ostringstream 汇;
    汇 << 流.rdbuf();
    std::string 原文 = 汇.str();

    if (!原文.empty() && static_cast<unsigned char>(原文[0]) == 0xEF) {
        原因 = "源文件带 UTF-8 BOM，glslang 会在第 0 行报非法字符，请存为无 BOM";
        return false;
    }

    定义.源路径 = 路径;
    定义.算子名 = std::filesystem::path(路径).stem().string();

    std::string 去注释 = 规范化文本(原文);
    std::string 展开 = 展开包含(路径, 去注释, 定义);
    return 解析布局(展开, 定义, 原因);
}

inline std::uint32_t 着色编译器::计算指纹(const std::vector<std::uint32_t>& 字) {
    std::uint32_t 指纹 = 2166136261u;                 // FNV-1a 32 位，与注册表一致
    for (std::uint32_t 字值 : 字) { 指纹 ^= 字值; 指纹 *= 16777619u; }
    return 指纹;
}

// 删临时产物：删不掉也不能抛。残留的 .tmp.spv 不会被 扫描 当成有效产物，
// 而抛异常会让整批编译在一个已经被杀软占用的文件上全线停摆。
inline void 删中间件(const std::string& 路径) {
    std::error_code 忽略;
    std::filesystem::remove(路径, 忽略);
}

inline bool 着色编译器::调用编译器(const 编译选项& 选项, const std::string& 源,
                                   const std::string& 输出, std::string& 日志) {
    // 先出临时 SPIR-V，再按需优化到最终路径，避免优化失败留下半成品
    std::string 中间 = 输出 + ".tmp.spv";
    std::ostringstream 命令;
    // 外层这对引号不是装饰：_popen 走 cmd /c，而 cmd 见到首字符是引号就会把首尾引号
    // 各削掉一个，于是可执行文件路径的收尾引号被吃掉，报"文件名、目录名或卷标语法不正确"。
    命令 << "\"\"" << 选项.编译器路径 << "\" -V --target-env " << 选项.目标环境
         << " -S comp -o \"" << 中间 << "\" \"" << 源 << "\" 2>&1\"";

    std::FILE* 管道 = VKTS_POPEN(命令.str().c_str(), "r");
    if (管道 == nullptr) {
        日志 = "无法启动编译器: " + 选项.编译器路径 + "（检查 PATH 或 编译选项.编译器路径）";
        return false;
    }
    char 缓冲[512];
    while (std::fgets(缓冲, sizeof(缓冲), 管道)) 日志 += 缓冲;
    int 退出码 = VKTS_PCLOSE(管道);
    if (退出码 != 0) {
        删中间件(中间);
        if (日志.empty()) 日志 = "编译器返回 " + std::to_string(退出码) + "，无输出";
        return false;
    }

    // 中间 文件必须在这里彻底关掉：Windows 上句柄不放，后面的 remove/rename 一律
    // 抛 "另一个程序正在使用此文件"，整批编译跟着一起挂。
    std::vector<char> 字节;
    {
        std::ifstream 读(中间, std::ios::binary);
        if (!读) { 日志 = "编译器报告成功但没有产出文件"; return false; }
        字节.assign((std::istreambuf_iterator<char>(读)), std::istreambuf_iterator<char>());
    }
    if (字节.size() % 4 != 0) { 删中间件(中间); 日志 = "产出长度不是 4 的倍数"; return false; }

    std::vector<std::uint32_t> 字(字节.size() / 4);
    std::memcpy(字.data(), 字节.data(), 字节.size());
    if (字.empty() || (字[0] != 0x07230203u && 字[0] != 0x03022307u)) {
        删中间件(中间);
        日志 = "产出的不是合法 SPIR-V";
        return false;
    }

    if (选项.优化工具路径.empty()) {
        std::error_code 错;
        std::filesystem::remove(输出, 错);       // rename 不覆盖只读/占用中的旧产物
        std::filesystem::rename(中间, 输出, 错);
        if (错) {
            // 退路：直接按字节写最终产物，中间件留着也无害
            std::ofstream 备(输出, std::ios::binary | std::ios::trunc);
            备.write(字节.data(), static_cast<std::streamsize>(字节.size()));
            删中间件(中间);
            if (!备) { 日志 = "无法写出产物: " + 输出; return false; }
        }
        return true;
    }
    std::ostringstream 优化;
    优化 << "\"\"" << 选项.优化工具路径 << "\" -O \"" << 中间 << "\" -o \"" << 输出
         << "\" 2>&1\"";
    管道 = VKTS_POPEN(优化.str().c_str(), "r");
    std::string 优化日志;
    bool 优化成功 = false;
    if (管道 != nullptr) {
        while (std::fgets(缓冲, sizeof(缓冲), 管道)) 优化日志 += 缓冲;
        优化成功 = (VKTS_PCLOSE(管道) == 0);
    }
    删中间件(中间);
    if (!优化成功) {
        // 降级：优化器不可用就用未优化产物，但必须在诊断里写清楚
        std::ofstream 备(输出, std::ios::binary | std::ios::trunc);
        备.write(字节.data(), static_cast<std::streamsize>(字节.size()));
        日志 += "[spirv-opt 未生效，已使用未优化产物] " + 优化日志;
    }
    return true;
}

inline void 着色编译器::写元信息头(const 算子定义& 定义) {
    std::ofstream 头(定义.头路径, std::ios::binary | std::ios::trunc);
    头 << "// ===========================================================================\n"
       << "//  自动生成自 " << 定义.源路径 << "  —— 请勿手工编辑\n"
       << "//  由 着色编译器 产出，与同目录 " << 定义.算子名 << ".spv 严格同源\n"
       << "// ===========================================================================\n"
       << "#pragma once\n\n"
       << "#include \"流水线调度器.hpp\"\n\n"
       << "namespace vk张量系统 {\n\n"
       << "inline const 着色器元信息 元信息_" << 定义.算子名 << " = [] {\n"
       << "    着色器元信息 信息;\n"
       << "    信息.算子名       = \"" << 定义.算子名 << "\";\n"
       << "    信息.工作组大小   = " << 定义.工作组大小 << "u;   // local_size_x\n"
       << "    信息.绑定数       = " << 定义.绑定数 << "u;   // " << 定义.绑定清单.size()
       << " 项含推送常量\n"
       << "    信息.推送常量字节 = " << 定义.推送常量字节 << "u;\n"
       << "    信息.产物指纹     = " << 定义.产物指纹 << "u;   // SPIR-V FNV-1a\n"
       << "    流水线调度器::获取().登记元信息(信息);\n"
       << "    return 信息;\n"
       << "}();\n\n"
       << "} // namespace vk张量系统\n";
}

inline void 着色编译器::写失败头(const 算子定义& 定义) {
    // 关键：让残留的旧产物在编译宿主时就炸，而不是运行时静默错绑
    std::string 消息 = 定义.诊断;
    std::replace(消息.begin(), 消息.end(), '"', '\'');
    消息.erase(std::remove(消息.begin(), 消息.end(), '\n'), 消息.end());
    std::ofstream 头(定义.头路径, std::ios::binary | std::ios::trunc);
    头 << "// 自动生成 —— 编译或解析失败，产物不可用\n"
       << "#pragma once\n"
       << "#error \"" << 定义.算子名 << ": " << 消息 << "\"\n";
}

inline std::vector<算子定义> 着色编译器::编译全部(编译选项 选项) {
    std::vector<算子定义> 结果;
    成功数_ = 0;
    诊断汇总_.clear();

    std::error_code 错误;
    std::filesystem::path 根(选项.源目录);
    if (!std::filesystem::exists(根, 错误)) throw std::runtime_error("源目录不存在: " + 选项.源目录);
    std::filesystem::create_directories(选项.产物目录, 错误);
    if (错误) throw std::runtime_error("产物目录创建失败: " + 选项.产物目录);

    std::vector<std::filesystem::path> 源清单;
    auto 收集 = [&](const std::filesystem::path& 路径) {
        std::string 后缀 = 路径.extension().string();
        if (后缀 == ".glsl" || 后缀 == ".comp" || 后缀 == ".geom" || 后缀 == ".frag") {
            源清单.push_back(路径);
        }
    };
    if (选项.递归) {
        for (std::filesystem::recursive_directory_iterator 它(根, 错误), 尾; 它 != 尾; 它.increment(错误)) {
            if (错误) break;
            if (它->is_regular_file()) 收集(它->path());
        }
    } else {
        for (const auto& 项 : std::filesystem::directory_iterator(根, 错误)) {
            if (错误) break;
            if (项.is_regular_file()) 收集(项.path());
        }
    }
    std::sort(源清单.begin(), 源清单.end());

    for (const auto& 源 : 源清单) {
        算子定义 定义;
        定义.算子名 = 源.stem().string();
        定义.产物路径 = (std::filesystem::path(选项.产物目录) / (定义.算子名 + ".spv")).string();
        定义.头路径 = 定义.产物路径 + ".hpp";

        if (!是spirv二进制(源.string()) && !含入口函数(源.string())) {
            定义.跳过 = true;
            定义.诊断 = "公共源，无入口函数，未编译";
            // 上一次的残留必须清掉，否则它会继续被 扫描 当成有效产物注册
            std::error_code 清;
            std::filesystem::remove(定义.产物路径, 清);
            std::filesystem::remove(定义.头路径, 清);
            结果.push_back(std::move(定义));
            continue;
        }

        bool 已是二进制 = 选项.允许源已是SPIRV && 是spirv二进制(源.string());
        if (已是二进制) {
            std::ifstream 读(源, std::ios::binary);
            std::vector<char> 字节((std::istreambuf_iterator<char>(读)), std::istreambuf_iterator<char>());
            std::vector<std::uint32_t> 字(字节.size() / 4);
            std::memcpy(字.data(), 字节.data(), 字节.size());
            定义.产物指纹 = 计算指纹(字);
            std::ofstream 写(定义.产物路径, std::ios::binary);
            写.write(字节.data(), static_cast<std::streamsize>(字节.size()));
            定义.成功 = false;   // 仍需要 .glsl 源来解析元信息
            定义.诊断 = "只找到已编译的 .spv，缺少可读源，无法解析绑定布局";
        } else {
            std::string 原因;
            if (!解析源文件(源.string(), 定义, 原因)) {
                定义.成功 = false;
                定义.诊断 = 源.string() + ": " + 原因;
            } else {
                std::string 日志;
                if (!调用编译器(选项, 源.string(), 定义.产物路径, 日志)) {
                    定义.成功 = false;
                    定义.诊断 = 日志.empty() ? 源.string() + ": 编译失败" : 日志;
                } else {
                    std::ifstream 读(定义.产物路径, std::ios::binary);
                    std::vector<char> 字节((std::istreambuf_iterator<char>(读)),
                                           std::istreambuf_iterator<char>());
                    std::vector<std::uint32_t> 字(字节.size() / 4);
                    std::memcpy(字.data(), 字节.data(), 字节.size());
                    定义.产物指纹 = 计算指纹(字);
                    定义.成功 = true;
                    定义.诊断 = 日志;   // 可能带 spirv-opt 降级提示
                    ++成功数_;
                }
            }
        }

        if (定义.成功) 写元信息头(定义); else 写失败头(定义);
        诊断汇总_ += (定义.成功 ? "[ok]   " : "[fail] ") + 定义.算子名 + ": " + 定义.诊断 + "\n";
        结果.push_back(std::move(定义));
    }

    if (选项.失败即抛出) {
        std::string 未成;
        for (const auto& 定义 : 结果) {
            if (!定义.成功 && !定义.跳过) 未成 += 定义.算子名 + "\n  ";
        }
        if (!未成.empty()) {
            throw std::runtime_error("着色编译失败：\n  " + 未成 + "\n" + 诊断汇总_);
        }
    }
    return 结果;
}

} // namespace vk张量系统
