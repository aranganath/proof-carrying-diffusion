# Proof Sidecar Diffusion

This repo trains a decoupled bounding-box/proof sidecar for text-to-image diffusion.
The Stable Diffusion image trajectory is frozen and unchanged. A separate proof
latent reads detached UNet features and predicts phrase-aligned heatmaps and boxes.

## What this implements

The image denoiser runs as usual:

```text
z_{t-1} = F_theta(z_t, prompt, t)
```

The proof sidecar is parallel only:

```text
u_{t-1}, boxes = G_phi(u_t, stopgrad(UNetFeatures_t), phrase_embeddings, t)
```

`u_t` and the predicted boxes are **not** fed back into the image latent. This is
therefore appropriate for an interpretability/probing paper: the sidecar reads
what the frozen generator exposes during denoising.

## Install

```bash
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended.

## Dataset

The code uses Flickr30k Entities annotations and boxes. The annotation repo is
cloned automatically. Images can be supplied locally via `--image-root`, or a
Hugging Face Flickr30k mirror can be materialized automatically.

For a quick first run:

```bash
python train.py
```

This will:

1. clone the Flickr30k Entities annotations,
2. materialize a capped number of Flickr30k images from Hugging Face,
3. build jsonl indexes for train/val,
4. load a frozen Stable Diffusion backbone,
5. train only the proof sidecar.

For a fuller run, remove the convenience caps:

```bash
python train.py \
  --max-hf-images 0 \
  --max-train-samples 0 \
  --max-val-samples 0 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --epochs 3
```

If you already have official Flickr30k images locally:

```bash
python train.py --image-root /path/to/flickr30k/images --no-use-hf-images
```

## Sampling

After training:

```bash
python sample.py \
  --checkpoint runs/proof_sidecar/checkpoint_last.pt \
  --prompt "A young girl's face looking through leaves" \
  --phrases "young girl's face|leaves" \
  --output-dir samples/leaves
```

Outputs:

- `samples/leaves/image.png`
- `samples/leaves/image_with_sidecar_boxes.png`

The image path is normal DDIM sampling. The box sidecar is parallel and non-intervening.

## Files

- `train.py`: end-to-end sidecar training.
- `sample.py`: normal image sampling plus sidecar boxes.
- `scripts/download_flickr30k.py`: dataset preparation only.
- `pcd/dataset.py`: Flickr30k Entities parsing, image materialization, heatmap targets.
- `pcd/models.py`: proof sidecar network.
- `pcd/sd_backbone.py`: frozen Stable Diffusion loader and feature hooks.
- `pcd/losses.py`: heatmap, diffusion-noise, objectness, and box losses.

## Notes

- Official Flickr30k images have usage terms. If the Hugging Face mirror is not
  available in your environment, use `--image-root` with a local image directory.
- The default model is `runwayml/stable-diffusion-v1-5`. If your Hugging Face
  account needs to accept a model license, do that before running.
- This is designed as a research scaffold, not a heavily optimized trainer.
