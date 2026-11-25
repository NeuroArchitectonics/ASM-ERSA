#ifndef ASM_TRAINER_H
#define ASM_TRAINER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#ifdef ASM_TRAINER_BUILD
#define ASM_TRAINER_API __declspec(dllexport)
#else
#define ASM_TRAINER_API __declspec(dllimport)
#endif
#else
#define ASM_TRAINER_API __attribute__((visibility("default")))
#endif

#define ASM_TRAINER_VERSION "1.0.0"
#define ASM_TRAINER_DEBUG_BF16 1u
#define ASM_TRAINER_DEBUG_MOE  2u

typedef struct AsmTrainerHandle AsmTrainerHandle;

typedef struct AsmTrainerConfig {
    uint32_t struct_size;
    uint32_t model;          /* 0 = 3B */
    uint32_t seed;
    uint32_t optimizer;      /* 0 = apollo_mini */
    uint32_t param_mode;     /* 0 = int8 direct, 1 = fp32 master */
    uint32_t grad_mode;      /* 0 = bf16, 1 = fp32 */
    uint32_t bf16_scope;     /* 0 = all, 1 = linear, 2 = moe */
    float learning_rate;
    float mini_scale;
    uint32_t keep_moe_master;
    uint32_t old_state_f32;
    uint32_t word_grad_f32;
    uint32_t debug_flags;
    uint32_t reserved[8];
} AsmTrainerConfig;

typedef struct AsmTrainerStats {
    uint32_t struct_size;
    uint64_t step;
    float loss;
    double forward_ms;
    double backward_ms;
    double update_ms;
    uint64_t vk_live_bytes;
    uint64_t bf16_params;
    uint64_t bf16_elems;
    uint64_t bf16_nonfinite;
    double bf16_global_norm;
    uint64_t reserved[4];
} AsmTrainerStats;

ASM_TRAINER_API AsmTrainerHandle* asm_trainer_create(const char* spv_dir,
                                                     const AsmTrainerConfig* config,
                                                     char* error_buffer,
                                                     size_t error_buffer_size);
ASM_TRAINER_API int asm_trainer_train_step(AsmTrainerHandle* handle,
                                           AsmTrainerStats* stats,
                                           char* error_buffer,
                                           size_t error_buffer_size);
ASM_TRAINER_API int asm_trainer_get_stats(AsmTrainerHandle* handle,
                                          AsmTrainerStats* stats,
                                          char* error_buffer,
                                          size_t error_buffer_size);
ASM_TRAINER_API void asm_trainer_destroy(AsmTrainerHandle* handle);
ASM_TRAINER_API const char* asm_trainer_version(void);

#ifdef __cplusplus
}
#endif

#endif
