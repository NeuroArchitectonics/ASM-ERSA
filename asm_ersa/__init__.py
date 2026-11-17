"""
ASM-ERSA: Addressable Slot Memory + Exact-Recall Sparse Attention
"""

from .asm_ersa import ASM_ERSA, ASM_ERSA_Layer, ERSA, GatedResidual, MoE, TransformerBlock
from .asm_ersa_dynamic_sparse_embed import DynamicSparseEmbedding
from .asm_ersa_omni import (
    ASM_ERSA_Omni, ERSA_Omni, ModalityMoE, OmniTransformerBlock,
    AudioEncoder, VisionEncoder, AudioOutputHead, ModalityEmbedding,
    Modality,
)

__version__ = "4.0.0"
__all__ = [
    "ASM_ERSA",
    "ASM_ERSA_Layer",
    "ERSA",
    "GatedResidual",
    "MoE",
    "TransformerBlock",
    "DynamicSparseEmbedding",
    "ASM_ERSA_Omni",
    "ERSA_Omni",
    "ModalityMoE",
    "OmniTransformerBlock",
    "AudioEncoder",
    "VisionEncoder",
    "AudioOutputHead",
    "ModalityEmbedding",
    "Modality",
]
