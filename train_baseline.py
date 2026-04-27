"""Stage 0 baseline trainer.

Runs either an AdamW (full BF16) or QLoRA (4-bit + LoRA) baseline on a
causal-LM corpus. Records loss, step time, tokens/sec, and peak GPU memory.

Usage:
    python train_baseline.py --config configs/baseline_adamw.yaml
    python train_baseline.py --config configs/baseline_qlora.yaml

Comparison against HMT (later stages) is the purpose of these baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from hmt.data import DataConfig, build_dataloader
from hmt.model_loader import load_baseline_adamw, load_baseline_qlora
from hmt.profiler import TrainingProfiler, format_step_stats, reset_peak_memory


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: DictConfig):
    baseline = cfg.baseline
    if baseline == "adamw":
        return load_baseline_adamw(
            model_name=cfg.model.name,
            dtype=cfg.model.dtype,
            gradient_checkpointing=cfg.model.get("gradient_checkpointing", False),
        )
    if baseline == "qlora":
        lora = cfg.model.lora
        return load_baseline_qlora(
            model_name=cfg.model.name,
            lora_r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            bnb_compute_dtype=cfg.model.bnb_compute_dtype,
            gradient_checkpointing=cfg.model.get("gradient_checkpointing", True),
        )
    raise ValueError(f"Unknown baseline '{baseline}'. Expected 'adamw' or 'qlora'.")


def lr_lambda(step: int, warmup: int, total: int, schedule: str) -> float:
    if step < warmup:
        return step / max(1, warmup)
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 1.0 - progress)
    raise ValueError(f"Unknown lr_schedule '{schedule}'")


def train(cfg: DictConfig) -> None:
    device = pick_device()
    print(f"[init] device={device}  baseline={cfg.baseline}  model={cfg.model.name}")

    loaded = build_model(cfg)
    model, tokenizer = loaded.model, loaded.tokenizer
    print(
        f"[init] params: trainable={loaded.trainable_params:,} "
        f"total={loaded.total_params:,} "
        f"({100 * loaded.trainable_params / max(1, loaded.total_params):.2f}%)"
    )

    if cfg.baseline == "adamw":
        # QLoRA path already places weights on GPU via device_map; only move for adamw.
        model.to(device)

    data_cfg = DataConfig(
        dataset=cfg.data.dataset,
        subset=cfg.data.get("subset"),
        split=cfg.data.split,
        text_field=cfg.data.text_field,
        seq_length=cfg.data.seq_length,
        streaming=cfg.data.streaming,
        shuffle_buffer=cfg.data.shuffle_buffer,
        seed=cfg.data.seed,
    )
    loader = build_dataloader(data_cfg, tokenizer, batch_size=cfg.training.batch_size)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(
        trainable,
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.betas),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )

    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w", encoding="utf-8")

    profiler = TrainingProfiler()
    reset_peak_memory()
    model.train()

    grad_accum = cfg.training.gradient_accumulation_steps
    max_steps = cfg.training.max_steps
    warmup = cfg.training.warmup_steps
    schedule = cfg.training.lr_schedule
    base_lr = cfg.training.learning_rate
    max_grad_norm = cfg.training.max_grad_norm
    log_interval = cfg.logging.log_interval

    step = 0
    micro = 0
    accum_loss = 0.0
    accum_tokens = 0

    profiler.start_step()
    optim.zero_grad(set_to_none=True)

    data_iter = iter(loader)
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, labels=labels)
        # Average loss across the accumulation window so the gradient scale
        # matches a single large batch.
        loss = out.loss / grad_accum
        loss.backward()

        accum_loss += out.loss.detach().float().item()
        accum_tokens += input_ids.numel()
        micro += 1

        if micro % grad_accum != 0:
            continue

        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)

        scale = lr_lambda(step, warmup, max_steps, schedule)
        for g in optim.param_groups:
            g["lr"] = base_lr * scale

        optim.step()
        optim.zero_grad(set_to_none=True)

        step += 1
        avg_loss = accum_loss / grad_accum
        stats = profiler.end_step(step=step, loss=avg_loss, tokens=accum_tokens)
        metrics_f.write(json.dumps({
            "step": stats.step,
            "loss": stats.loss,
            "lr": base_lr * scale,
            "step_time_s": stats.step_time_s,
            "tokens_per_s": stats.tokens_per_s,
            "cur_mem_mb": stats.cur_mem_mb,
            "peak_mem_mb": stats.peak_mem_mb,
        }) + "\n")
        metrics_f.flush()

        if step % log_interval == 0 or step == 1:
            print(format_step_stats(stats))

        accum_loss = 0.0
        accum_tokens = 0
        if step < max_steps:
            profiler.start_step()

    metrics_f.close()
    print(f"[done] run peak GPU memory: {profiler.run_peak_mb():.1f} MB")
    print(f"[done] metrics written to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. training.max_steps=50",
    )
    args = parser.parse_args()

    base = OmegaConf.load(args.config)
    if args.overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(args.overrides))

    # Echo resolved config for reproducibility.
    print("[config]\n" + OmegaConf.to_yaml(base))
    train(base)


if __name__ == "__main__":
    main()
