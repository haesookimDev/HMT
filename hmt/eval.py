"""Evaluation: cross-entropy and perplexity on a held-out split.

Computes per-token NLL across `max_batches` of the eval split, weights each
batch by its true token count, and returns mean loss + exp(loss) as PPL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from hmt.data import DataConfig, build_dataloader


@dataclass
class EvalResult:
    loss: float
    ppl: float
    tokens: int
    batches: int


@torch.no_grad()
def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    data_cfg: DataConfig,
    *,
    device: torch.device,
    batch_size: int = 1,
    max_batches: int = 50,
) -> EvalResult:
    loader = build_dataloader(data_cfg, tokenizer, batch_size=batch_size)

    was_training = model.training
    model.eval()

    loss_sum = 0.0
    n_tokens = 0
    n_batches = 0

    for batch in loader:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(input_ids=input_ids, labels=labels)
        # HF mean loss is over (seq_len - 1) shifted positions per sample.
        n_tok = labels.shape[0] * (labels.shape[1] - 1)
        loss_sum += out.loss.detach().float().item() * n_tok
        n_tokens += n_tok
        n_batches += 1

    if was_training:
        model.train()

    if n_tokens == 0:
        raise RuntimeError("Eval loader produced 0 tokens; check data.eval_split / dataset.")

    mean_loss = loss_sum / n_tokens
    # Clamp before exp to avoid overflow for very-poorly-initialized models.
    ppl = math.exp(min(mean_loss, 20.0))
    return EvalResult(loss=mean_loss, ppl=ppl, tokens=n_tokens, batches=n_batches)
