from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from diffusers import DDIMScheduler

from pcd.losses import cxcywh_to_xyxy
from pcd.models import ProofSidecarDenoiser
from pcd.sd_backbone import (
    UNetFeatureTap,
    encode_phrase_batch,
    encode_prompts,
    load_stable_diffusion_components,
)
from pcd.utils import ensure_dir, get_dtype, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an image normally and add proof boxes with a decoupled sidecar.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--phrases", type=str, default=None, help="Phrase slots separated by '|'. Default: the whole prompt as one phrase.")
    p.add_argument("--pretrained-model", type=str, default=None, help="Defaults to checkpoint training model.")
    p.add_argument("--output-dir", type=str, default="./samples")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    return p.parse_args()


def decode_latents(vae, latents: torch.Tensor, scaling_factor: float) -> Image.Image:
    with torch.no_grad():
        image = vae.decode(latents / scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().cpu().permute(1, 2, 0).float().numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def draw_boxes(image: Image.Image, boxes_cxcywh: torch.Tensor, phrases: List[str], scores: torch.Tensor | None = None) -> Image.Image:
    im = image.copy()
    draw = ImageDraw.Draw(im)
    W, H = im.size
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh.detach().cpu().float()).numpy()
    palette = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "blue"]
    for i, phrase in enumerate(phrases):
        x1, y1, x2, y2 = boxes_xyxy[i]
        x1, x2 = x1 * W, x2 * W
        y1, y2 = y1 * H, y2 * H
        color = palette[i % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = phrase
        if scores is not None:
            label = f"{phrase} {float(scores[i]):.2f}"
        tx, ty = x1 + 3, max(0, y1 - 16)
        draw.rectangle([tx - 1, ty, tx + 8 * len(label) + 6, ty + 15], fill=color)
        draw.text((tx + 2, ty + 1), label, fill="black")
    return im


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    pretrained_model = args.pretrained_model or train_args.get("pretrained_model", "runwayml/stable-diffusion-v1-5")
    hook_names = ckpt.get("hook_names") or train_args.get("hook_names", "down_blocks.1,down_blocks.2,mid_block,up_blocks.1").split(",")
    heatmap_size = int(ckpt.get("heatmap_size", train_args.get("heatmap_size", 64)))
    phrase_dim = int(ckpt.get("phrase_dim", 768))

    phrases = [p.strip() for p in (args.phrases.split("|") if args.phrases else [args.prompt]) if p.strip()]
    if not phrases:
        phrases = [args.prompt]
    K = len(phrases)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(args.mixed_precision)
    if device.type != "cuda":
        dtype = torch.float32

    comps = load_stable_diffusion_components(pretrained_model, device, dtype, scheduler_type="ddim")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    scheduler.set_timesteps(args.num_steps, device=device)
    feature_tap = UNetFeatureTap(comps.unet, hook_names=hook_names, detach=True)

    sidecar = ProofSidecarDenoiser(
        phrase_dim=phrase_dim,
        num_feature_levels=len(hook_names),
        heatmap_size=heatmap_size,
        hidden_dim=256,
    ).to(device)
    sidecar.load_state_dict(ckpt["sidecar"], strict=True)
    sidecar.eval()

    prompt_embeds = encode_prompts(comps.tokenizer, comps.text_encoder, [args.prompt], device, dtype)
    uncond_embeds = encode_prompts(comps.tokenizer, comps.text_encoder, [""], device, dtype)
    phrase_embeds = encode_phrase_batch(comps.tokenizer, comps.text_encoder, [phrases], device, torch.float32)

    latent_h = args.height // 8
    latent_w = args.width // 8
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn((1, comps.unet.config.in_channels, latent_h, latent_w), generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    u_t = torch.randn((1, K, heatmap_size, heatmap_size), generator=generator, device=device, dtype=torch.float32)

    last_out = None
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="sampling"):
            if args.guidance_scale > 1.0:
                latent_model_input = torch.cat([latents, latents], dim=0)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                text_input = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                t_batch = torch.tensor([int(t), int(t)], device=device, dtype=torch.long)
                noise_pred_all, features_all = feature_tap(latent_model_input, t_batch, text_input)
                noise_uncond, noise_text = noise_pred_all.chunk(2)
                noise_pred = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)
                features = [f[1:2].float() for f in features_all]
            else:
                latent_model_input = scheduler.scale_model_input(latents, t)
                t_batch = torch.tensor([int(t)], device=device, dtype=torch.long)
                noise_pred, features = feature_tap(latent_model_input, t_batch, prompt_embeds)
                features = [f.float() for f in features]

            # The normal image trajectory is updated exactly as in DDIM. The proof sidecar only reads features.
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            proof_t = torch.tensor([int(t)], device=device, dtype=torch.long)
            last_out = sidecar(features, u_t, phrase_embeds, proof_t)
            # Lightweight proof-latent update. This is not fed back to the image latents.
            # DDIM with the same t schedule is enough for qualitative proof sampling.
            alpha_prod_t = scheduler.alphas_cumprod[int(t)].to(device=device, dtype=u_t.dtype)
            beta_prod_t = 1 - alpha_prod_t
            pred_x0 = (u_t - beta_prod_t.sqrt() * last_out["noise_pred"]) / alpha_prod_t.sqrt().clamp_min(1e-6)
            u_t = pred_x0.clamp(-1.5, 1.5)

    image = decode_latents(comps.vae, latents, comps.latent_scaling_factor)
    assert last_out is not None
    boxes = last_out["boxes_cxcywh"][0]
    scores = torch.sigmoid(last_out["obj_logits"])[0]
    boxed = draw_boxes(image, boxes, phrases, scores=scores)

    image.save(out_dir / "image.png")
    boxed.save(out_dir / "image_with_sidecar_boxes.png")
    print(f"Saved {out_dir / 'image.png'}")
    print(f"Saved {out_dir / 'image_with_sidecar_boxes.png'}")

    feature_tap.remove()


if __name__ == "__main__":
    main()
