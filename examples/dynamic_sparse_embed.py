"""
Example: Dynamic Sparse Embedding (tiny config, runs on CPU in seconds).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from asm_ersa import DynamicSparseEmbedding

embed = DynamicSparseEmbedding(
    vocab_size=1000,
    d_model=64,
    max_ngram_entries=1000,
    cache_size=1000,
    ngram_order=2,
    use_cache=True,
)

params = embed.get_param_count()
print("Dynamic Sparse Embedding Parameters:")
print(f"  Base Embed:    {params['base_embed'] / 1e6:.2f}M")
print(f"  N-gram Embed:  {params['ngram_embed'] / 1e6:.2f}M")
print(f"  Combine Proj:  {params['combine_proj'] / 1e6:.2f}M")
print(f"  Gate Proj:     {params['gate_proj'] / 1e6:.2f}M")
print(f"  Total:         {params['total'] / 1e6:.2f}M")

batch_size, seq_len = 2, 16
token_ids = torch.randint(0, 1000, (batch_size, seq_len))
context_ids = torch.randint(0, 1000, (batch_size, seq_len, 2))

with torch.no_grad():
    out = embed(token_ids, context_ids=context_ids)

print(f"\nOutput shape: {tuple(out.shape)}")
print("✓ Forward pass successful")
