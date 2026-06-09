from __future__ import annotations

from contextlib import nullcontext
from typing import List, Sequence

import torch
from torch import nn
from transformers import AutoModel

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional until the training image installs peft
    LoraConfig = None
    TaskType = None
    get_peft_model = None


class DINOv3FeatureExtractor(nn.Module):
    """Extracts spatial feature maps from selected DINOv3 hidden states."""

    def __init__(
        self,
        checkpoint_path: str,
        output_layers: Sequence[int] = (6, 12, 18, 24),
        freeze: bool = True,
        drop_path_rate: float = 0.0,
        unfreeze_last_n_blocks: int = 0,
        lora: dict | None = None,
    ) -> None:
        super().__init__()
        self.output_layers = tuple(output_layers)
        self.freeze = freeze
        self.unfreeze_last_n_blocks = max(0, int(unfreeze_last_n_blocks))
        self.lora_config = dict(lora or {})
        self.lora_enabled = bool(self.lora_config.get("enabled", False))
        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        if drop_path_rate > 0 and hasattr(self.model.config, "drop_path_rate"):
            self.model.config.drop_path_rate = drop_path_rate
            if hasattr(self.model, "set_drop_path_rate"):
                self.model.set_drop_path_rate(drop_path_rate)

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.has_cls_token = True
        self.total_blocks = len(self.model.layer) if hasattr(self.model, "layer") else 0
        self.fully_frozen = self.freeze and self.unfreeze_last_n_blocks == 0 and not self.lora_enabled

        if self.lora_enabled:
            if self.unfreeze_last_n_blocks > 0:
                raise ValueError("LoRA and `unfreeze_last_n_blocks` should not be enabled at the same time.")
            self._apply_lora()
        elif freeze:
            self.model.requires_grad_(False)
            if self.unfreeze_last_n_blocks > 0:
                self._unfreeze_last_blocks(self.unfreeze_last_n_blocks)
            if self.fully_frozen:
                self.model.eval()

    def _unfreeze_last_blocks(self, count: int) -> None:
        if not hasattr(self.model, "layer"):
            raise AttributeError("DINOv3 backbone does not expose `layer`; cannot partially unfreeze blocks.")
        total_blocks = len(self.model.layer)
        start = max(0, total_blocks - count)
        for index in range(start, total_blocks):
            self.model.layer[index].requires_grad_(True)
        if hasattr(self.model, "norm"):
            self.model.norm.requires_grad_(True)

    def _apply_lora(self) -> None:
        if LoraConfig is None or TaskType is None or get_peft_model is None:
            raise ImportError("LoRA is enabled but `peft` is not installed in the current environment.")
        if not hasattr(self.model, "layer"):
            raise AttributeError("DINOv3 backbone does not expose `layer`; cannot attach LoRA by block index.")
        self.model.requires_grad_(False)
        train_last_n_blocks = int(self.lora_config.get("train_last_n_blocks", 4))
        start = max(0, self.total_blocks - train_last_n_blocks)
        attention_modules = self.lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        target_modules = [
            f"layer.{block_index}.attention.{module_name}"
            for block_index in range(start, self.total_blocks)
            for module_name in attention_modules
        ]
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(self.lora_config.get("rank", 16)),
            lora_alpha=int(self.lora_config.get("alpha", 32)),
            lora_dropout=float(self.lora_config.get("dropout", 0.05)),
            bias=self.lora_config.get("bias", "none"),
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        self.fully_frozen = False

    def save_lora(self, output_dir: str) -> None:
        if self.lora_enabled and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)

    def trainable_parameter_counts(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return total, trainable

    def train(self, mode: bool = True) -> "DINOv3FeatureExtractor":
        super().train(mode)
        if self.fully_frozen:
            self.model.eval()
        else:
            self.model.train(mode)
        return self

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        context = torch.no_grad() if self.fully_frozen else nullcontext()
        with context:
            outputs = self.model(pixel_values=inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        assert hidden_states is not None

        batch_size, _, height, width = inputs.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        prefix_tokens = int(self.has_cls_token) + self.num_register_tokens

        feature_maps: List[torch.Tensor] = []
        for layer_index in self.output_layers:
            tokens = hidden_states[layer_index]
            patch_tokens = tokens[:, prefix_tokens : prefix_tokens + patch_height * patch_width, :]
            feature = patch_tokens.transpose(1, 2).reshape(
                batch_size,
                self.hidden_size,
                patch_height,
                patch_width,
            )
            feature_maps.append(feature)
        return feature_maps
