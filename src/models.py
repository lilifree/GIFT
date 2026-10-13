"""Model loading, teacher/student setup, EMA management."""

import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel


def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_target_modules: Optional[list[str]] = None,
    gradient_checkpointing: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
):
    """Load model and tokenizer.

    Follows verl-style loading: rank 0 loads weights on CPU,
    other ranks init with empty weights, then FSDP/DDP syncs.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine attn implementation (flash_attention_2 -> sdpa fallback)
    try:
        _test = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _test_kwargs = dict(
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        # Quick check if flash_attention_2 is available
        AutoModelForCausalLM.from_config(_test, **{k: v for k, v in _test_kwargs.items() if k != 'trust_remote_code'})
    except (ImportError, ValueError):
        print(f"Warning: {attn_implementation} not available, falling back to sdpa")
        attn_implementation = "sdpa"

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    if world_size > 1:
        # verl-style: rank 0 loads on CPU, others wait then load
        # This avoids 8 processes hammering disk simultaneously
        if rank == 0:
            print("Rank 0: loading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        dist.barrier()
        if rank != 0:
            print(f"Rank {rank}: loading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        dist.barrier()
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        except (ImportError, ValueError):
            print(f"Warning: {attn_implementation} not available, falling back to sdpa")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )

    if use_lora:
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled")

    return model, tokenizer


class EMATeacher:
    """Exponential Moving Average teacher model."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(model)

    def _init_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights: shadow = decay * shadow + (1-decay) * model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """Temporarily replace model weights with EMA weights."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore original model weights after teacher forward pass."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class TeacherContext:
    """Context manager for teacher forward passes with different strategies."""

    def __init__(
        self,
        model: nn.Module,
        teacher_mode: str = "dynamic",  # "dynamic", "fixed", "ema", "snapshot"
        ema_teacher: Optional[EMATeacher] = None,
    ):
        self.model = model
        self.teacher_mode = teacher_mode
        self.ema_teacher = ema_teacher

    def __enter__(self):
        if self.teacher_mode in ("ema", "snapshot") and self.ema_teacher is not None:
            self.ema_teacher.apply_shadow(self.model)
        elif self.teacher_mode == "fixed" and isinstance(self.model, PeftModel):
            self.model.disable_adapter_layers()
        return self

    def __exit__(self, *args):
        if self.teacher_mode in ("ema", "snapshot") and self.ema_teacher is not None:
            self.ema_teacher.restore(self.model)
        elif self.teacher_mode == "fixed" and isinstance(self.model, PeftModel):
            self.model.enable_adapter_layers()
