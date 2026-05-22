from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import DDPMScheduler

from pcd.dataset import (
    Flickr30kProofDataset,
    collate_proof_batch,
    prepare_flickr30k_entities,
)
from pcd.losses import (
    masked_bce_with_logits,
    masked_box_loss,
    masked_mse,
    objectness_loss,
    pairwise_iou_aligned,
    cxcywh_to_xyxy,
    soft_dice_loss_with_logits,
)
from pcd.models import ProofSidecarDenoiser
from pcd.sd_backbone import (
    DEFAULT_HOOKS,
    UNetFeatureTap,
    encode_images_to_latents,
    encode_phrase_batch,
    encode_prompts,
    load_stable_diffusion_components,
)
from pcd.utils import count_parameters, ensure_dir, get_dtype, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a decoupled proof-sidecar on Flickr30k Entities.")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--image-root", type=str, default=None, help="Local Flickr30k image directory. If omitted, HF images are materialized.")
    p.add_argument("--use-hf-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-hf-images", type=int, default=2500, help="Convenience cap for first run. Use 0 for all images.")
    p.add_argument("--max-index-samples", type=int, default=0, help="Cap jsonl indexing. Use 0 for all indexed samples.")
    p.add_argument("--max-train-samples", type=int, default=2000)
    p.add_argument("--max-val-samples", type=int, default=200)

    p.add_argument("--pretrained-model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output-dir", type=str, default="./runs/proof_sidecar")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--heatmap-size", type=int, default=64)
    p.add_argument("--max-phrases", type=int, default=8)
    p.add_argument("--hook-names", type=str, default=",".join(DEFAULT_HOOKS))

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=25)

    p.add_argument("--lambda-noise", type=float, default=1.0)
    p.add_argument("--lambda-bce", type=float, default=1.0)
    p.add_argument("--lambda-dice", type=float, default=1.0)
    p.add_argument("--lambda-box", type=float, default=1.0)
    p.add_argument("--lambda-obj", type=float, default=0.25)
    return p.parse_args()


def _optional_int(value: int) -> int | None:
    return None if value is None or value <= 0 else value


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def make_noisy_proof(heatmaps: torch.Tensor, scheduler: DDPMScheduler) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert heatmaps [0,1] to proof latents [-1,1] and add DDPM noise."""
    B = heatmaps.shape[0]
    device = heatmaps.device
    u0 = heatmaps * 2.0 - 1.0
    noise = torch.randn_like(u0)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
    u_t = scheduler.add_noise(u0, noise, timesteps)
    return u_t, noise, timesteps


def forward_backbone_and_sidecar(
    batch: Dict,
    comps,
    feature_tap: UNetFeatureTap,
    sidecar: ProofSidecarDenoiser,
    proof_scheduler: DDPMScheduler,
    device: torch.device,
    backbone_dtype: torch.dtype,
    train_sidecar: bool = True,
) -> Dict[str, torch.Tensor]:
    pixel_values = batch["pixel_values"].to(device=device, dtype=backbone_dtype)
    heatmaps = batch["heatmaps"].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = encode_images_to_latents(comps.vae, pixel_values, comps.latent_scaling_factor)
        noise = torch.randn_like(latents)
        img_timesteps = torch.randint(
            0,
            comps.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = comps.scheduler.add_noise(latents, noise, img_timesteps)
        prompt_embeds = encode_prompts(comps.tokenizer, comps.text_encoder, batch["captions"], device, backbone_dtype)
        _, features = feature_tap(noisy_latents, img_timesteps, prompt_embeds)
        # Keep the interpretation branch detached from image generation.
        features = [f.detach() for f in features]
        phrase_embeds = encode_phrase_batch(comps.tokenizer, comps.text_encoder, batch["phrase_texts"], device, torch.float32)

    u_t, proof_noise, proof_timesteps = make_noisy_proof(heatmaps, proof_scheduler)
    out = sidecar(features, u_t, phrase_embeds, proof_timesteps)
    out["proof_noise_target"] = proof_noise
    out["proof_timesteps"] = proof_timesteps
    return out


def compute_losses(out: Dict[str, torch.Tensor], batch: Dict, device: torch.device, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    heatmaps = batch["heatmaps"].to(device=device, dtype=out["heat_logits"].dtype)
    valid = batch["valid"].to(device=device, dtype=out["heat_logits"].dtype)
    target_cxcywh = batch["boxes_cxcywh"].to(device=device, dtype=out["boxes_cxcywh"].dtype)
    target_xyxy = batch["boxes_xyxy"].to(device=device, dtype=out["boxes_cxcywh"].dtype)

    loss_noise = masked_mse(out["noise_pred"], out["proof_noise_target"].to(out["noise_pred"].dtype), valid)
    loss_bce = masked_bce_with_logits(out["heat_logits"], heatmaps, valid)
    loss_dice = soft_dice_loss_with_logits(out["heat_logits"], heatmaps, valid)
    loss_box = masked_box_loss(out["boxes_cxcywh"], target_cxcywh, target_xyxy, valid)
    loss_obj = objectness_loss(out["obj_logits"], valid)
    total = (
        args.lambda_noise * loss_noise
        + args.lambda_bce * loss_bce
        + args.lambda_dice * loss_dice
        + args.lambda_box * loss_box
        + args.lambda_obj * loss_obj
    )
    return {
        "loss": total,
        "loss_noise": loss_noise,
        "loss_bce": loss_bce,
        "loss_dice": loss_dice,
        "loss_box": loss_box,
        "loss_obj": loss_obj,
    }


def box_iou_metric(out: Dict[str, torch.Tensor], batch: Dict, device: torch.device) -> torch.Tensor:
    valid = batch["valid"].to(device=device, dtype=torch.float32)
    target_xyxy = batch["boxes_xyxy"].to(device=device, dtype=torch.float32)
    pred_xyxy = cxcywh_to_xyxy(out["boxes_cxcywh"].float())
    iou = pairwise_iou_aligned(pred_xyxy, target_xyxy)
    return (iou * valid).sum() / valid.sum().clamp_min(1.0)


def save_checkpoint(path: Path, sidecar: ProofSidecarDenoiser, optimizer, step: int, args: argparse.Namespace, phrase_dim: int, hook_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sidecar": sidecar.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "args": vars(args),
            "phrase_dim": phrase_dim,
            "hook_names": hook_names,
            "heatmap_size": args.heatmap_size,
            "max_phrases": args.max_phrases,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    save_json(vars(args), out_dir / "args.json")

    max_hf_images = _optional_int(args.max_hf_images)
    max_index_samples = _optional_int(args.max_index_samples)
    train_index, image_root, _ = prepare_flickr30k_entities(
        args.data_root,
        split="train",
        image_root=args.image_root,
        use_hf_images=args.use_hf_images,
        max_index_samples=max_index_samples,
        max_hf_images=max_hf_images,
    )
    val_index, _, _ = prepare_flickr30k_entities(
        args.data_root,
        split="val",
        image_root=image_root,
        use_hf_images=False,
        max_index_samples=max_index_samples,
    )

    train_ds = Flickr30kProofDataset(
        train_index,
        image_root=image_root,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        max_phrases=args.max_phrases,
        max_samples=_optional_int(args.max_train_samples),
    )
    val_ds = Flickr30kProofDataset(
        val_index,
        image_root=image_root,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        max_phrases=args.max_phrases,
        max_samples=_optional_int(args.max_val_samples),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_proof_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_proof_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_dtype = get_dtype(args.mixed_precision)
    if device.type != "cuda":
        backbone_dtype = torch.float32

    print(f"Loading frozen diffusion backbone: {args.pretrained_model}")
    comps = load_stable_diffusion_components(args.pretrained_model, device, backbone_dtype, scheduler_type="ddpm")
    hook_names = [x.strip() for x in args.hook_names.split(",") if x.strip()]
    feature_tap = UNetFeatureTap(comps.unet, hook_names=hook_names, detach=True)
    phrase_dim = int(comps.text_encoder.config.hidden_size)

    sidecar = ProofSidecarDenoiser(
        phrase_dim=phrase_dim,
        num_feature_levels=len(hook_names),
        heatmap_size=args.heatmap_size,
        hidden_dim=256,
    ).to(device)
    proof_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # LazyConv2d layers are initialized by a dry forward pass before the optimizer is built.
    print("Initializing sidecar lazy layers ...")
    first_batch = next(iter(train_loader))
    with torch.no_grad(), autocast_context(device, backbone_dtype):
        _ = forward_backbone_and_sidecar(first_batch, comps, feature_tap, sidecar, proof_scheduler, device, backbone_dtype)
    print(f"Trainable sidecar parameters: {count_parameters(sidecar):,}")

    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.mixed_precision == "fp16"))

    global_step = 0
    sidecar.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            with autocast_context(device, backbone_dtype):
                out = forward_backbone_and_sidecar(batch, comps, feature_tap, sidecar, proof_scheduler, device, backbone_dtype)
                losses = compute_losses(out, batch, device, args)
                loss = losses["loss"] / args.grad_accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(sidecar.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    with torch.no_grad():
                        iou = box_iou_metric(out, batch, device).item()
                    log = {k: float(v.detach().float().item()) for k, v in losses.items()}
                    log["box_iou"] = iou
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in log.items() if k in ["loss", "loss_box", "box_iou"]})
                    print({"step": global_step, **log})

                if global_step % args.save_every == 0:
                    save_checkpoint(out_dir / f"checkpoint_{global_step}.pt", sidecar, optimizer, global_step, args, phrase_dim, hook_names)

        # A tiny validation pass each epoch.
        sidecar.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for vbatch in tqdm(val_loader, desc="validation"):
                with autocast_context(device, backbone_dtype):
                    vout = forward_backbone_and_sidecar(vbatch, comps, feature_tap, sidecar, proof_scheduler, device, backbone_dtype)
                    vlosses = compute_losses(vout, vbatch, device, args)
                val_loss_sum += float(vlosses["loss"].detach().float().item())
                val_iou_sum += float(box_iou_metric(vout, vbatch, device).item())
                val_n += 1
        print({"epoch": epoch + 1, "val_loss": val_loss_sum / max(val_n, 1), "val_box_iou": val_iou_sum / max(val_n, 1)})
        sidecar.train()
        save_checkpoint(out_dir / "checkpoint_last.pt", sidecar, optimizer, global_step, args, phrase_dim, hook_names)

    feature_tap.remove()
    print(f"Done. Last checkpoint: {out_dir / 'checkpoint_last.pt'}")


if __name__ == "__main__":
    main()
