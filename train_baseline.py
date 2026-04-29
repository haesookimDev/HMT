"""Unified trainer for Stage 0 baselines and Stage 1+ HMT runs.

Dispatches on ``training.optimizer``:
  - ``adamw``          : full BF16 / fp32 AdamW (Stage 0 baseline)
  - ``lowrank_adamw``  : Stage 1 GaLore-style low-rank AdamW
  - QLoRA is selected via ``baseline: qlora`` and uses ``adamw`` internally.

Records loss, step time, tokens/sec, and peak GPU memory; periodically
evaluates perplexity on the validation split.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from hmt.data import DataConfig, build_dataloader
from hmt.eval import evaluate_perplexity
from hmt.model_loader import load_baseline_adamw, load_baseline_qlora
from hmt.autograd import patch_model_int8_linear
from hmt.memory import ActivationPolicy
from hmt.optim import (
    ApolloAdamW,
    EnergyRankScheduler,
    LowRankAdamW,
    attach_projectors_from_grads,
    refresh_projectors_from_grads,
    select_target_params,
)
from hmt.profiler import TrainingProfiler, format_step_stats, reset_peak_memory
from hmt.utils import seed_everything


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
    seed = int(cfg.get("seed", cfg.data.seed))
    seed_everything(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = pick_device()
    print(f"[init] device={device}  baseline={cfg.baseline}  model={cfg.model.name}  seed={seed}")

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

    eval_cfg = cfg.get("eval", None)
    eval_interval = int(eval_cfg.interval) if eval_cfg is not None else 0
    eval_data_cfg = None
    if eval_interval > 0:
        eval_data_cfg = DataConfig(
            dataset=cfg.data.dataset,
            subset=cfg.data.get("subset"),
            split=eval_cfg.split,
            text_field=cfg.data.text_field,
            seq_length=cfg.data.seq_length,
            streaming=False,  # deterministic order; small validation set fits in RAM
            shuffle_buffer=cfg.data.shuffle_buffer,
            seed=cfg.data.seed,
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_name = cfg.training.get("optimizer", "adamw")
    lowrank_ctx = None
    if opt_name == "adamw":
        optim = AdamW(
            trainable,
            lr=cfg.training.learning_rate,
            betas=tuple(cfg.training.betas),
            eps=cfg.training.eps,
            weight_decay=cfg.training.weight_decay,
        )
    elif opt_name == "lowrank_adamw":
        lr_cfg = cfg.training.lowrank
        optim = LowRankAdamW(
            trainable,
            lr=cfg.training.learning_rate,
            betas=tuple(cfg.training.betas),
            eps=cfg.training.eps,
            weight_decay=cfg.training.weight_decay,
        )
        targets = select_target_params(model, pattern=lr_cfg.target_pattern)
        if not targets:
            raise RuntimeError(
                f"target_pattern '{lr_cfg.target_pattern}' matched no nn.Linear modules. "
                "Inspect model.named_modules() and adjust the pattern."
            )

        sched_cfg = lr_cfg.get("rank_scheduler", None)
        if sched_cfg is not None:
            scheduler = EnergyRankScheduler(
                candidates=tuple(int(c) for c in sched_cfg.candidates),
                threshold=float(sched_cfg.threshold),
            )
            rank_arg = None
            rank_desc = (
                f"scheduler(τ={sched_cfg.threshold}, "
                f"candidates={list(scheduler.candidates)})"
            )
        else:
            scheduler = None
            rank_arg = int(lr_cfg.rank)
            rank_desc = f"rank={rank_arg}"

        method = str(lr_cfg.get("method", "full"))
        print(
            f"[init] lowrank_adamw: {len(targets)} target Linear modules, "
            f"mode={lr_cfg.mode} {rank_desc} method={method} K={lr_cfg.basis_update_interval}"
        )
        lowrank_ctx = {
            "targets": targets,
            "mode": lr_cfg.mode,
            "rank": rank_arg,
            "scheduler": scheduler,
            "method": method,
            "K": int(lr_cfg.basis_update_interval),
            "attached": False,
        }
    elif opt_name == "apollo_adamw":
        apollo_cfg = cfg.training.apollo
        optim = ApolloAdamW(
            trainable,
            lr=cfg.training.learning_rate,
            betas=tuple(cfg.training.betas),
            eps=cfg.training.eps,
            weight_decay=cfg.training.weight_decay,
            scaling=str(apollo_cfg.scaling),
        )
        print(f"[init] apollo_adamw: scaling={apollo_cfg.scaling}")
    else:
        raise ValueError(f"Unknown training.optimizer: {opt_name}")

    # Stage 3: activation memory policy. Patch nn.Linear → CompressedLinear
    # AFTER select_target_params has already captured weight references for
    # the low-rank optimizer (parameter identity is preserved through patching).
    act_cfg = cfg.get("activation_policy", None)
    if act_cfg is not None:
        policy = ActivationPolicy.from_config(act_cfg)
        patched = patch_model_int8_linear(model, policy)
        if patched:
            print(
                f"[init] activation compression: patched {len(patched)} Linear modules "
                f"to INT8 (block_size={policy.block_size})"
            )
        else:
            print("[init] activation compression: policy matched no nn.Linear modules")

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

        if lowrank_ctx is not None:
            if not lowrank_ctx["attached"]:
                attached = attach_projectors_from_grads(
                    optim, lowrank_ctx["targets"],
                    mode=lowrank_ctx["mode"],
                    rank=lowrank_ctx["rank"],
                    scheduler=lowrank_ctx["scheduler"],
                    method=lowrank_ctx["method"],
                )
                lowrank_ctx["attached"] = True
                rank_log = {name: int(proj.rank) for name, proj in attached.items()}
                if rank_log:
                    avg = sum(rank_log.values()) / len(rank_log)
                    mn = min(rank_log.values())
                    mx = max(rank_log.values())
                    print(
                        f"[init] attached projectors to {len(attached)} layers, "
                        f"rank avg={avg:.1f} min={mn} max={mx}"
                    )
                    (out_dir / "ranks.json").write_text(json.dumps({
                        "ranks": rank_log,
                        "avg": avg, "min": mn, "max": mx,
                    }, indent=2))
            elif step > 0 and step % lowrank_ctx["K"] == 0:
                refresh_projectors_from_grads(optim, lowrank_ctx["targets"])

        optim.step()
        optim.zero_grad(set_to_none=True)

        step += 1
        avg_loss = accum_loss / grad_accum
        stats = profiler.end_step(step=step, loss=avg_loss, tokens=accum_tokens)
        metrics_f.write(json.dumps({
            "event": "train",
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

        if eval_interval > 0 and step % eval_interval == 0:
            er = evaluate_perplexity(
                model=model,
                tokenizer=tokenizer,
                data_cfg=eval_data_cfg,
                device=device,
                batch_size=cfg.training.batch_size,
                max_batches=int(eval_cfg.max_batches),
            )
            print(
                f"[eval ] step {step:>6d} | loss {er.loss:7.4f} | ppl {er.ppl:9.3f} | "
                f"tokens {er.tokens} | batches {er.batches}"
            )
            metrics_f.write(json.dumps({
                "event": "eval",
                "step": step,
                "eval_loss": er.loss,
                "eval_ppl": er.ppl,
                "eval_tokens": er.tokens,
                "eval_batches": er.batches,
            }) + "\n")
            metrics_f.flush()

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
