"""C++ 分词器（`分词器入口.hpp`）的 ctypes 封装。

使用前需要把封装编译成 DLL，并设置：
    ASM_ERSA_TOKENIZER_DLL   DLL 路径
    ASM_ERSA_TOKENIZER       分词器模型目录（含 vocab.json / tokenizer.json）

训练脚本在设置了 `ASM_ERSA_TOKENIZER_DLL` 时会优先使用本类，
不再走 HuggingFace tokenizer。
"""

import ctypes
import os

import torch


class CppTokenizer:
    """把 `便捷分词器` 的 编码/词表大小 暴露成 HF tokenizer 的最小兼容接口。"""

    def __init__(self, dll_path: str, model_dir: str):
        self._lib = ctypes.CDLL(os.path.abspath(dll_path))

        self._lib.tok_create.restype = ctypes.c_void_p
        self._lib.tok_create.argtypes = [ctypes.c_wchar_p]
        self._lib.tok_vocab_size.restype = ctypes.c_int
        self._lib.tok_vocab_size.argtypes = [ctypes.c_void_p]
        self._lib.tok_encode.restype = ctypes.c_int
        self._lib.tok_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ]
        self._lib.tok_free.argtypes = [ctypes.c_void_p]

        self._handle = self._lib.tok_create(model_dir)
        if not self._handle:
            raise RuntimeError(f"加载 C++ 分词器失败：{model_dir}")
        self.vocab_size = self._lib.tok_vocab_size(self._handle)

    def encode(self, text: str):
        data = (text or "").encode("utf-8")
        n = self._lib.tok_encode(self._handle, data, None, 0)
        if n <= 0:
            return []
        buf = (ctypes.c_int * n)()
        self._lib.tok_encode(self._handle, data, buf, n)
        return list(buf)

    def __call__(self, text, truncation: bool = True, max_length: int = 2048,
                 return_tensors=None, **kwargs):
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                self._lib.tok_free(self._handle)
                self._handle = None
        except Exception:
            pass
