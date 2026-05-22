from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


DEFAULT_HOOKS = ["down_blocks.1", "down_blocks.2", "mid_block", "up_blocks.1"]


@dataclass
class StableDiffusionComponents:
    tokenizer: object
    text_encoder: nn.Module
    vae: nn.Module
    unet: nn.Module
    scheduler: object
    latent_scaling_factor: float


def load_stable_diffusion_components(
    pretrained_model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    scheduler_type: str = "ddpm",
) -> StableDiffusionComponents:
    """Load frozen Stable Diffusion components from diffusers."""
    from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    if scheduler_type.lower() == "ddim":
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    else:
        scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder.eval().requires_grad_(False).to(device, dtype=dtype)
    vae.eval().requires_grad_(False).to(device, dtype=dtype)
    unet.eval().requires_grad_(False).to(device, dtype=dtype)

    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    return StableDiffusionComponents(tokenizer, text_encoder, vae, unet, scheduler, float(scaling))


def _extract_tensor_from_hook_output(output) -> Optional[torch.Tensor]:
    if torch.is_tensor(output) and output.ndim == 4:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 4:
                return item
            if isinstance(item, (tuple, list)):
                for sub in item:
                    if torch.is_tensor(sub) and sub.ndim == 4:
                        return sub
    if hasattr(output, "sample") and torch.is_tensor(output.sample) and output.sample.ndim == 4:
        return output.sample
    return None


class UNetFeatureTap:
    """Forward-hook based feature reader for a diffusers UNet.

    The hooks do not alter the UNet output. They only copy selected feature
    tensors so a sidecar can read them.
    """

    def __init__(self, unet: nn.Module, hook_names: Sequence[str] = DEFAULT_HOOKS, detach: bool = True) -> None:
        self.unet = unet
        self.hook_names = list(hook_names)
        self.detach = bool(detach)
        self._features: Dict[str, torch.Tensor] = {}
        self._handles = []
        modules = dict(unet.named_modules())
        missing = [name for name in self.hook_names if name not in modules]
        if missing:
            available = [k for k in modules.keys() if k.startswith(("down_blocks", "mid_block", "up_blocks"))]
            raise ValueError(
                f"Hook names not found: {missing}. Some available names: {available[:30]}"
            )
        for name in self.hook_names:
            module = modules[name]
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            tensor = _extract_tensor_from_hook_output(output)
            if tensor is None:
                return
            self._features[name] = tensor.detach() if self.detach else tensor
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(self, latents: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor):
        self._features = {}
        out = self.unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states)
        sample = out.sample if hasattr(out, "sample") else out[0]
        features = [self._features[name] for name in self.hook_names]
        return sample, features


def encode_prompts(tokenizer, text_encoder, prompts: Sequence[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tokens = tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = getattr(tokens, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        output = text_encoder(input_ids, attention_mask=attention_mask)
    return output.last_hidden_state.to(dtype)


def encode_phrase_batch(tokenizer, text_encoder, phrase_texts: Sequence[Sequence[str]], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode phrase slots as pooled CLIP embeddings.

    Empty phrase slots become empty-string embeddings; the valid mask from the
    dataset decides whether their losses are active.
    """
    B = len(phrase_texts)
    K = len(phrase_texts[0]) if B > 0 else 0
    flat = [p if p else "" for row in phrase_texts for p in row]
    tokens = tokenizer(
        flat,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        output = text_encoder(input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.reshape(B, K, -1).to(dtype)


def encode_images_to_latents(vae: nn.Module, pixel_values: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    with torch.no_grad():
        dist = vae.encode(pixel_values).latent_dist
        latents = dist.sample() * scaling_factor
    return latents
