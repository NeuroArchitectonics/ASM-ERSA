from __future__ import annotations

import argparse
import os
from pathlib import Path

from asm_trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="ASM-ERSA 3B Vulkan trainer shared-library example")
    default_library = Path(__file__).resolve().parent / (
        "asm_trainer.dll" if os.name == "nt" else "libasm_ersa_trainer.so"
    )
    parser.add_argument("--spv-dir", type=Path, default=Path(__file__).resolve().parent / "spv")
    parser.add_argument("--library", type=Path, default=default_library)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--debug-bf16", action="store_true")
    parser.add_argument("--save", type=Path, default=None,
                        help="保存检查点路径：写 ASM8BIN 权重和同名 .opt 优化器状态")
    parser.add_argument("--resume", type=Path, default=None,
                        help="加载检查点路径，恢复权重/优化器状态后继续训练")
    args = parser.parse_args()

    with Trainer(
        args.spv_dir,
        library=args.library,
        seed=args.seed,
        learning_rate=args.lr,
        debug_bf16=args.debug_bf16,
    ) as trainer:
        print(f"ASM trainer {trainer.version}")
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)
            print(f"resumed from {args.resume} step={trainer.get_stats().step}")
        for _ in range(args.steps):
            stats = trainer.train_step()
            print(
                f"step={stats.step} loss={stats.loss:.6f} "
                f"fwd={stats.forward_ms:.2f}ms bwd={stats.backward_ms:.2f}ms "
                f"upd={stats.update_ms:.2f}ms vk={stats.vk_live_bytes / (1 << 30):.2f}GiB "
                f"bf16_nonfinite={stats.bf16_nonfinite}"
            )
            latest = trainer.get_stats()
            print(f"get_stats step={latest.step} loss={latest.loss:.6f}")
        if args.save is not None:
            trainer.save_checkpoint(args.save)
            print(f"checkpoint -> {args.save} (+ .opt)")


if __name__ == "__main__":
    main()
