// ===========================================================================
//  自动生成自 ASM-ERSA-Design/vk/glsl_dev/apollo_mini_stat.comp  —— 请勿手工编辑
//  由 着色编译器 产出，与同目录 apollo_mini_stat.spv 严格同源
// ===========================================================================
#pragma once

#include "流水线调度器.hpp"

namespace vk张量系统 {

inline const 着色器元信息 元信息_apollo_mini_stat = [] {
    着色器元信息 信息;
    信息.算子名       = "apollo_mini_stat";
    信息.工作组大小   = 256u;   // local_size_x
    信息.绑定数       = 4u;   // 5 项含推送常量
    信息.推送常量字节 = 32u;
    信息.产物指纹     = 41831701u;   // SPIR-V FNV-1a
    流水线调度器::获取().登记元信息(信息);
    return 信息;
}();

} // namespace vk张量系统
