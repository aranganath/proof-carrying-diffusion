from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / max(half, 1))
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ProofSidecarDenoiser(nn.Module):
    """A decoupled proof-latent sidecar for frozen diffusion UNet features.

    The image generator is never conditioned on the proof latent. This module
    only reads detached image-diffusion features and predicts phrase-aligned
    proof heatmaps/boxes.
    """

    def __init__(
        self,
        phrase_dim: int,
        num_feature_levels: int,
        heatmap_size: int = 64,
        hidden_dim: int = 256,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.phrase_dim = int(phrase_dim)
        self.num_feature_levels = int(num_feature_levels)
        self.heatmap_size = int(heatmap_size)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)

        self.feature_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LazyConv2d(hidden_dim, kernel_size=1),
                    nn.GroupNorm(num_groups=32 if hidden_dim % 32 == 0 else 8, num_channels=hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(num_feature_levels)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(num_groups=32 if hidden_dim % 32 == 0 else 8, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.u_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.GroupNorm(num_groups=32 if hidden_dim % 32 == 0 else 8, num_channels=hidden_dim),
            nn.SiLU(),
        )
        self.phrase_mlp = nn.Sequential(
            nn.Linear(phrase_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(num_groups=32 if hidden_dim % 32 == 0 else 8, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2, 1),  # [proof_noise, clean_heatmap_logit]
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _fuse_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_feature_levels:
            raise ValueError(f"Expected {self.num_feature_levels} feature levels, got {len(features)}")
        out = None
        for feat, proj in zip(features, self.feature_projs):
            if feat.ndim != 4:
                raise ValueError(f"Expected feature tensor [B,C,H,W], got shape {tuple(feat.shape)}")
            f = proj(feat)
            f = F.interpolate(f, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)
            out = f if out is None else out + f
        assert out is not None
        return self.fuse(out)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        u_t: torch.Tensor,
        phrase_embeds: torch.Tensor,
        proof_timesteps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: list of detached UNet feature maps, each [B,C,H,W].
            u_t: noisy proof latent, [B,K,Hm,Wm], usually target_heatmap*2-1 noised by a scheduler.
            phrase_embeds: CLIP phrase embeddings, [B,K,D].
            proof_timesteps: scheduler timesteps, [B].
        Returns:
            noise_pred: [B,K,Hm,Wm]
            heat_logits: [B,K,Hm,Wm]
            boxes_cxcywh: [B,K,4] in normalized coordinates
            obj_logits: [B,K]
        """
        B, K, Hm, Wm = u_t.shape
        if Hm != self.heatmap_size or Wm != self.heatmap_size:
            raise ValueError(f"u_t must have spatial size {self.heatmap_size}; got {(Hm, Wm)}")
        base = self._fuse_features(features)  # [B,H,Hm,Wm]
        t_emb = sinusoidal_timestep_embedding(proof_timesteps, self.time_dim).to(base.dtype)
        t_bias = self.time_mlp(t_emb).view(B, self.hidden_dim, 1, 1)
        base = base + t_bias

        phrase_bias = self.phrase_mlp(phrase_embeds.to(base.dtype)).view(B, K, self.hidden_dim, 1, 1)
        x = base[:, None, :, :, :] + phrase_bias
        x = x.reshape(B * K, self.hidden_dim, Hm, Wm)

        u = u_t.reshape(B * K, 1, Hm, Wm).to(x.dtype)
        x = x + self.u_proj(u)

        spatial = self.spatial_head(x)
        noise_pred = spatial[:, 0].reshape(B, K, Hm, Wm)
        heat_logits = spatial[:, 1].reshape(B, K, Hm, Wm)

        # Attention-pool features for each phrase using the predicted heatmap.
        attn = heat_logits.reshape(B * K, 1, Hm * Wm)
        attn = torch.softmax(attn, dim=-1)
        flat = x.reshape(B * K, self.hidden_dim, Hm * Wm)
        pooled = (flat * attn).sum(dim=-1)
        boxes_cxcywh = torch.sigmoid(self.box_head(pooled)).reshape(B, K, 4)
        obj_logits = self.obj_head(pooled).reshape(B, K)

        return {
            "noise_pred": noise_pred,
            "heat_logits": heat_logits,
            "boxes_cxcywh": boxes_cxcywh,
            "obj_logits": obj_logits,
        }
