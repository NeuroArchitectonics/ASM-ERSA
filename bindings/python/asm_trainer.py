from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional


class Config(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("model", ctypes.c_uint32),
        ("seed", ctypes.c_uint32),
        ("optimizer", ctypes.c_uint32),
        ("param_mode", ctypes.c_uint32),
        ("grad_mode", ctypes.c_uint32),
        ("bf16_scope", ctypes.c_uint32),
        ("learning_rate", ctypes.c_float),
        ("mini_scale", ctypes.c_float),
        ("keep_moe_master", ctypes.c_uint32),
        ("old_state_f32", ctypes.c_uint32),
        ("word_grad_f32", ctypes.c_uint32),
        ("debug_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 8),
    ]


class Stats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("step", ctypes.c_uint64),
        ("loss", ctypes.c_float),
        ("forward_ms", ctypes.c_double),
        ("backward_ms", ctypes.c_double),
        ("update_ms", ctypes.c_double),
        ("vk_live_bytes", ctypes.c_uint64),
        ("bf16_params", ctypes.c_uint64),
        ("bf16_elems", ctypes.c_uint64),
        ("bf16_nonfinite", ctypes.c_uint64),
        ("bf16_global_norm", ctypes.c_double),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class TrainerError(RuntimeError):
    pass


def _default_library() -> Path:
    override = os.environ.get("ASM_TRAINER_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    name = "asm_trainer.dll" if os.name == "nt" else "libasm_ersa_trainer.so"
    return Path(__file__).resolve().parent / name


def _load_library(path: Path):
    if not path.is_file():
        raise TrainerError(f"找不到共享库: {path}")
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise TrainerError(f"加载共享库失败: {path}: {exc}") from exc

    lib.asm_trainer_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(Config),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.asm_trainer_create.restype = ctypes.c_void_p
    lib.asm_trainer_train_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Stats),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.asm_trainer_train_step.restype = ctypes.c_int
    lib.asm_trainer_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Stats),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.asm_trainer_get_stats.restype = ctypes.c_int
    lib.asm_trainer_destroy.argtypes = [ctypes.c_void_p]
    lib.asm_trainer_destroy.restype = None
    lib.asm_trainer_version.argtypes = []
    lib.asm_trainer_version.restype = ctypes.c_char_p
    return lib


class Trainer:
    def __init__(
        self,
        spv_dir: str | Path,
        *,
        library: str | Path | None = None,
        seed: int = 7,
        learning_rate: float = 1e-4,
        mini_scale: float = 128.0,
        param_mode: int = 0,
        grad_mode: int = 0,
        bf16_scope: int = 0,
        keep_moe_master: bool = False,
        old_state_f32: bool = False,
        word_grad_f32: bool = False,
        debug_bf16: bool = False,
        debug_moe: bool = False,
    ) -> None:
        self._lib = _load_library(Path(library) if library else _default_library())
        self._handle: Optional[int] = None
        self._error = ctypes.create_string_buffer(2048)
        config = Config(
            struct_size=ctypes.sizeof(Config),
            model=0,
            seed=seed,
            optimizer=0,
            param_mode=param_mode,
            grad_mode=grad_mode,
            bf16_scope=bf16_scope,
            learning_rate=learning_rate,
            mini_scale=mini_scale,
            keep_moe_master=int(keep_moe_master),
            old_state_f32=int(old_state_f32),
            word_grad_f32=int(word_grad_f32),
            debug_flags=(1 if debug_bf16 else 0) | (2 if debug_moe else 0),
        )
        spv_path = str(Path(spv_dir).expanduser().resolve()).encode("utf-8")
        handle = self._lib.asm_trainer_create(
            spv_path,
            ctypes.byref(config),
            self._error,
            ctypes.sizeof(self._error),
        )
        if not handle:
            message = self._error.value.decode("utf-8", errors="replace")
            raise TrainerError(message or "trainer 创建失败")
        self._handle = int(handle)

    @property
    def version(self) -> str:
        return self._lib.asm_trainer_version().decode("utf-8")

    def train_step(self) -> Stats:
        if self._handle is None:
            raise TrainerError("trainer 已关闭")
        stats = Stats(struct_size=ctypes.sizeof(Stats))
        rc = self._lib.asm_trainer_train_step(
            self._handle,
            ctypes.byref(stats),
            self._error,
            ctypes.sizeof(self._error),
        )
        if rc != 0:
            message = self._error.value.decode("utf-8", errors="replace")
            raise TrainerError(message or f"train_step 失败，rc={rc}")
        return stats

    def get_stats(self) -> Stats:
        if self._handle is None:
            raise TrainerError("trainer 已关闭")
        stats = Stats(struct_size=ctypes.sizeof(Stats))
        rc = self._lib.asm_trainer_get_stats(
            self._handle,
            ctypes.byref(stats),
            self._error,
            ctypes.sizeof(self._error),
        )
        if rc != 0:
            message = self._error.value.decode("utf-8", errors="replace")
            raise TrainerError(message or f"get_stats 失败，rc={rc}")
        return stats

    def close(self) -> None:
        if self._handle is not None:
            self._lib.asm_trainer_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
