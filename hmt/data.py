"""Dataset loading and sequence packing for causal LM training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DataConfig:
    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    seq_length: int = 1024
    streaming: bool = True
    shuffle_buffer: int = 1000
    seed: int = 42


class PackedCausalLMDataset(IterableDataset):
    """Concatenate tokenized text and slice into fixed-length sequences.

    No padding or attention masking — every position is a real token.
    Loss is the standard next-token CE; labels mirror input_ids.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        text_field: str = "text",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define an eos_token_id for packing.")
        self.eos_id = eos

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        target_len = self.seq_length + 1  # need one extra token for shift
        for example in self.hf_dataset:
            text = example.get(self.text_field)
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buffer.extend(ids)
            buffer.append(self.eos_id)

            while len(buffer) >= target_len:
                chunk = buffer[:target_len]
                buffer = buffer[target_len - 1:]  # overlap by 1 so labels align
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def build_dataloader(
    cfg: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    raw = load_dataset(
        cfg.dataset,
        cfg.subset,
        split=cfg.split,
        streaming=cfg.streaming,
    )
    if cfg.streaming:
        raw = raw.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)

    packed = PackedCausalLMDataset(
        hf_dataset=raw,
        tokenizer=tokenizer,
        seq_length=cfg.seq_length,
        text_field=cfg.text_field,
    )
    return DataLoader(
        packed,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
