"""Model loading for baselines.

Stage 0 supports two baselines:
  - "adamw": full BF16 model (no quantization, no adapters)
  - "qlora": 4-bit base + LoRA adapters (requires CUDA + bitsandbytes)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    trainable_params: int
    total_params: int


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def _count_params(model: PreTrainedModel) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # Causal LMs frequently lack a pad token; reuse EOS for batched padding.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_baseline_adamw(
    model_name: str,
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = False,
    device_map: Optional[str] = None,
) -> LoadedModel:
    """Load a full-precision (BF16) model for AdamW baseline."""
    torch_dtype = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # gradient checkpointing requires inputs to keep gradients enabled
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    tokenizer = _load_tokenizer(model_name)
    trainable, total = _count_params(model)
    return LoadedModel(model=model, tokenizer=tokenizer, trainable_params=trainable, total_params=total)


def load_baseline_qlora(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    bnb_compute_dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
) -> LoadedModel:
    """Load a 4-bit quantized base model with LoRA adapters (QLoRA).

    Requires CUDA + bitsandbytes. Will raise informatively if unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA baseline requires CUDA. bitsandbytes 4-bit kernels are CUDA-only. "
            "Use the AdamW baseline on non-CUDA systems."
        )
    try:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "QLoRA baseline needs `bitsandbytes` and `peft`. "
            "Install with: pip install '.[qlora]' peft"
        ) from e

    compute_dtype = _resolve_dtype(bnb_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )

    if target_modules is None:
        # Common targets for Llama-style architectures. Adjust per model family.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    tokenizer = _load_tokenizer(model_name)
    trainable, total = _count_params(model)
    return LoadedModel(model=model, tokenizer=tokenizer, trainable_params=trainable, total_params=total)
