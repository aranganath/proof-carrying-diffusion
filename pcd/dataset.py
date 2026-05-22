"""Flickr30k Entities data utilities for proof-sidecar training.

The dataset returns a full caption, a fixed number of phrase slots, a target
box per phrase slot, and a low-resolution heatmap per phrase slot. The image
path can come from either a local Flickr30k image directory or a Hugging Face
mirror. The official Flickr30k Entities repository contains the phrase/box
annotations and splits, but not the images.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


ENTITY_RE = re.compile(r"\[\/EN#(?P<cid>\d+)(?P<types>(?:\/[A-Za-z0-9_\-]+)*) (?P<phrase>[^\]]+)\]")


@dataclass
class PhraseAnn:
    chain_id: str
    phrase: str
    types: List[str]
    boxes: List[List[float]]  # xyxy in original image coordinates


def ensure_flickr30k_entities_repo(data_root: str | Path) -> Path:
    """Download the Flickr30k Entities annotation repo if it is not present."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    repo_dir = data_root / "flickr30k_entities"
    if (repo_dir / "Annotations").exists() and (repo_dir / "Sentences").exists():
        return repo_dir

    if repo_dir.exists() and not (repo_dir / ".git").exists():
        raise RuntimeError(
            f"{repo_dir} exists but does not look like the Flickr30k Entities repo. "
            "Move it or pass a different --data-root."
        )

    url = "https://github.com/BryanPlummer/flickr30k_entities.git"
    print(f"Cloning Flickr30k Entities annotations from {url} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(repo_dir)],
        check=True,
    )
    return repo_dir


def parse_sentence_line(line: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse one Flickr30k Entities sentence line.

    Returns a cleaned caption and phrase annotations. Not-visual phrases with
    chain id 0 are skipped later if they do not have boxes.
    """
    phrases: List[Dict[str, Any]] = []

    def repl(match: re.Match[str]) -> str:
        cid = match.group("cid")
        types = [t for t in match.group("types").split("/") if t]
        phrase = match.group("phrase").strip()
        phrases.append({"chain_id": cid, "types": types, "phrase": phrase})
        return phrase

    clean = ENTITY_RE.sub(repl, line.strip())
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean, phrases


def parse_xml_boxes(xml_path: str | Path) -> Tuple[Dict[str, List[List[float]]], Tuple[int, int]]:
    """Parse a Flickr30k Entities XML annotation file.

    Returns a mapping chain_id -> list of boxes and the original image size
    as (width, height).
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        width = int(float(size.findtext("width", default="0")))
        height = int(float(size.findtext("height", default="0")))
    else:
        width, height = 0, 0

    boxes_by_chain: Dict[str, List[List[float]]] = {}
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        for name in obj.findall("name"):
            cid = (name.text or "").strip()
            if cid and cid != "0":
                boxes_by_chain.setdefault(cid, []).append([xmin, ymin, xmax, ymax])
    return boxes_by_chain, (width, height)


def union_box(boxes: Sequence[Sequence[float]]) -> List[float]:
    arr = np.asarray(boxes, dtype=np.float32)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())]


def read_split_ids(repo_dir: str | Path, split: str) -> List[str]:
    repo_dir = Path(repo_dir)
    candidates = [repo_dir / f"{split}.txt", repo_dir / "Splits" / f"{split}.txt"]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip().replace(".jpg", "") for line in f if line.strip()]
    raise FileNotFoundError(f"Could not find split file for '{split}' in {repo_dir}")


def build_flickr30k_entities_index(
    repo_dir: str | Path,
    split: str,
    out_jsonl: str | Path,
    max_samples: Optional[int] = None,
) -> Path:
    """Build a jsonl index of caption-level phrase-grounding examples."""
    repo_dir = Path(repo_dir)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    image_ids = read_split_ids(repo_dir, split)
    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for image_id in tqdm(image_ids, desc=f"Indexing Flickr30k Entities/{split}"):
            xml_path = repo_dir / "Annotations" / f"{image_id}.xml"
            sent_path = repo_dir / "Sentences" / f"{image_id}.txt"
            if not xml_path.exists() or not sent_path.exists():
                continue
            boxes_by_chain, size = parse_xml_boxes(xml_path)
            if not boxes_by_chain:
                continue
            with open(sent_path, "r", encoding="utf-8") as f:
                for cap_idx, line in enumerate(f):
                    caption, phrase_infos = parse_sentence_line(line)
                    anns: List[Dict[str, Any]] = []
                    for info in phrase_infos:
                        cid = str(info["chain_id"])
                        if cid == "0" or cid not in boxes_by_chain:
                            continue
                        boxes = boxes_by_chain[cid]
                        anns.append(
                            {
                                "chain_id": cid,
                                "phrase": info["phrase"],
                                "types": info["types"],
                                "boxes": boxes,
                                "box_union": union_box(boxes),
                            }
                        )
                    if not anns:
                        continue
                    rec = {
                        "image_id": image_id,
                        "filename": f"{image_id}.jpg",
                        "caption_index": cap_idx,
                        "caption": caption,
                        "orig_size": [int(size[0]), int(size[1])],
                        "phrases": anns,
                    }
                    out.write(json.dumps(rec) + "\n")
                    count += 1
                    if max_samples is not None and count >= max_samples:
                        print(f"Wrote {count} examples to {out_jsonl}")
                        return out_jsonl
    print(f"Wrote {count} examples to {out_jsonl}")
    return out_jsonl


def materialize_hf_flickr30k_images(
    image_root: str | Path,
    split_names: Sequence[str] = ("train", "val", "test"),
    hf_name: str = "nlphuji/flickr30k",
    max_images: Optional[int] = None,
) -> Path:
    """Download/save Flickr30k images from a Hugging Face dataset mirror.

    This is provided as a convenience. If you have the official Flickr30k
    images locally, pass --image-root instead and skip this function.
    """
    from datasets import load_dataset

    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    print(f"Loading Hugging Face dataset {hf_name} ...")
    ds = load_dataset(hf_name)

    saved = 0
    for hf_split, dset in ds.items():
        if split_names and hf_split not in set(split_names):
            # Some HF mirrors provide one split with an internal 'split' column;
            # do not skip it unless it clearly is a different split.
            pass
        for row in tqdm(dset, desc=f"Saving HF images/{hf_split}"):
            filename = row.get("filename") or row.get("file_name") or row.get("image_id")
            if filename is None:
                # The nlphuji/flickr30k card exposes a filename column.
                continue
            filename = str(filename)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename = filename + ".jpg"
            out_path = image_root / filename
            if out_path.exists():
                continue
            image = row.get("image")
            if image is None:
                continue
            if isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                img = Image.open(image).convert("RGB")
            img.save(out_path, quality=95)
            saved += 1
            if max_images is not None and saved >= max_images:
                print(f"Saved {saved} images to {image_root}")
                return image_root
    print(f"Saved {saved} new images to {image_root}")
    return image_root


def prepare_flickr30k_entities(
    data_root: str | Path,
    split: str = "train",
    image_root: Optional[str | Path] = None,
    use_hf_images: bool = True,
    max_index_samples: Optional[int] = None,
    max_hf_images: Optional[int] = None,
) -> Tuple[Path, Path, Path]:
    """Ensure annotations, images, and an index file exist.

    Returns (index_jsonl, image_root, annotation_repo_dir).
    """
    data_root = Path(data_root)
    repo_dir = ensure_flickr30k_entities_repo(data_root)

    if image_root is None:
        image_root = data_root / "flickr30k_images"
    else:
        image_root = Path(image_root)

    if use_hf_images:
        has_any = image_root.exists() and any(image_root.glob("*.jpg"))
        if not has_any:
            materialize_hf_flickr30k_images(image_root, max_images=max_hf_images)

    index_path = data_root / "processed" / f"flickr30k_entities_{split}.jsonl"
    if not index_path.exists():
        build_flickr30k_entities_index(repo_dir, split, index_path, max_samples=max_index_samples)
    return index_path, Path(image_root), repo_dir


class Flickr30kProofDataset(Dataset):
    """Caption-level phrase grounding examples with fixed phrase slots."""

    def __init__(
        self,
        index_jsonl: str | Path,
        image_root: str | Path,
        image_size: int = 512,
        heatmap_size: int = 64,
        max_phrases: int = 8,
        max_samples: Optional[int] = None,
    ) -> None:
        self.index_jsonl = Path(index_jsonl)
        self.image_root = Path(image_root)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.max_phrases = int(max_phrases)

        self.records: List[Dict[str, Any]] = []
        with open(self.index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if (self.image_root / rec["filename"]).exists():
                    self.records.append(rec)
                if max_samples is not None and len(self.records) >= max_samples:
                    break
        if not self.records:
            raise RuntimeError(
                f"No usable records found in {self.index_jsonl}. Check that images exist in {self.image_root}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _resize_image_and_boxes(self, image: Image.Image, boxes_xyxy: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        orig_w, orig_h = image.size
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        if boxes_xyxy.size > 0:
            boxes_xyxy = boxes_xyxy.astype(np.float32).copy()
            boxes_xyxy[:, [0, 2]] *= self.image_size / max(float(orig_w), 1.0)
            boxes_xyxy[:, [1, 3]] *= self.image_size / max(float(orig_h), 1.0)
            boxes_xyxy = np.clip(boxes_xyxy, 0.0, float(self.image_size))
        return image, boxes_xyxy

    def _box_to_heatmap(self, box_xyxy: Sequence[float]) -> torch.Tensor:
        hm = torch.zeros((self.heatmap_size, self.heatmap_size), dtype=torch.float32)
        scale = self.heatmap_size / float(self.image_size)
        x1, y1, x2, y2 = [float(v) * scale for v in box_xyxy]
        ix1 = int(np.floor(np.clip(x1, 0, self.heatmap_size - 1)))
        iy1 = int(np.floor(np.clip(y1, 0, self.heatmap_size - 1)))
        ix2 = int(np.ceil(np.clip(x2, 1, self.heatmap_size)))
        iy2 = int(np.ceil(np.clip(y2, 1, self.heatmap_size)))
        if ix2 <= ix1:
            ix2 = min(ix1 + 1, self.heatmap_size)
        if iy2 <= iy1:
            iy2 = min(iy1 + 1, self.heatmap_size)
        hm[iy1:iy2, ix1:ix2] = 1.0
        return hm

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        path = self.image_root / rec["filename"]
        image = Image.open(path).convert("RGB")

        phrases = rec["phrases"][: self.max_phrases]
        boxes_orig = np.zeros((len(phrases), 4), dtype=np.float32)
        for i, ph in enumerate(phrases):
            boxes_orig[i] = np.asarray(ph["box_union"], dtype=np.float32)

        image, boxes_resized = self._resize_image_and_boxes(image, boxes_orig)
        pixel = TF.to_tensor(image) * 2.0 - 1.0

        K = self.max_phrases
        heatmaps = torch.zeros((K, self.heatmap_size, self.heatmap_size), dtype=torch.float32)
        boxes_xyxy = torch.zeros((K, 4), dtype=torch.float32)
        boxes_cxcywh = torch.zeros((K, 4), dtype=torch.float32)
        valid = torch.zeros((K,), dtype=torch.float32)
        phrase_texts = [""] * K
        phrase_types: List[List[str]] = [[] for _ in range(K)]

        for i, ph in enumerate(phrases):
            if i >= K:
                break
            x1, y1, x2, y2 = boxes_resized[i].tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy[i] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32) / float(self.image_size)
            cx = (x1 + x2) / 2.0 / float(self.image_size)
            cy = (y1 + y2) / 2.0 / float(self.image_size)
            bw = (x2 - x1) / float(self.image_size)
            bh = (y2 - y1) / float(self.image_size)
            boxes_cxcywh[i] = torch.tensor([cx, cy, bw, bh], dtype=torch.float32).clamp(0.0, 1.0)
            heatmaps[i] = self._box_to_heatmap([x1, y1, x2, y2])
            valid[i] = 1.0
            phrase_texts[i] = ph["phrase"]
            phrase_types[i] = ph.get("types", [])

        return {
            "pixel_values": pixel,
            "caption": rec["caption"],
            "filename": rec["filename"],
            "phrase_texts": phrase_texts,
            "phrase_types": phrase_types,
            "heatmaps": heatmaps,
            "boxes_xyxy": boxes_xyxy,
            "boxes_cxcywh": boxes_cxcywh,
            "valid": valid,
        }


def collate_proof_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "captions": [b["caption"] for b in batch],
        "filenames": [b["filename"] for b in batch],
        "phrase_texts": [b["phrase_texts"] for b in batch],
        "phrase_types": [b["phrase_types"] for b in batch],
        "heatmaps": torch.stack([b["heatmaps"] for b in batch], dim=0),
        "boxes_xyxy": torch.stack([b["boxes_xyxy"] for b in batch], dim=0),
        "boxes_cxcywh": torch.stack([b["boxes_cxcywh"] for b in batch], dim=0),
        "valid": torch.stack([b["valid"] for b in batch], dim=0),
    }
