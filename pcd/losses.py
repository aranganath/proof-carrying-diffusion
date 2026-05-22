from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss over valid phrase slots.

    logits, targets: [B, K, H, W]
    valid: [B, K] with 1 for valid phrase slots.
    """
    probs = torch.sigmoid(logits)
    valid = valid.float()
    dims = (-1, -2)
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    loss = 1.0 - dice
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss.mean(dim=(-1, -2))
    valid = valid.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = (pred - target).pow(2).mean(dim=(-1, -2))
    valid = valid.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp_min(0) * (boxes[..., 3] - boxes[..., 1]).clamp_min(0)


def pairwise_iou_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    return inter / union.clamp_min(eps)


def giou_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = pairwise_iou_aligned(boxes1, boxes2, eps=eps)
    lt_c = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    rb_c = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    wh_c = (rb_c - lt_c).clamp_min(0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    return iou - (area_c - union) / area_c.clamp_min(eps)


def masked_box_loss(
    pred_cxcywh: torch.Tensor,
    target_cxcywh: torch.Tensor,
    target_xyxy: torch.Tensor,
    valid: torch.Tensor,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> torch.Tensor:
    valid = valid.float()
    pred_cxcywh = pred_cxcywh.clamp(0.0, 1.0)
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)
    l1 = F.l1_loss(pred_cxcywh, target_cxcywh, reduction="none").sum(dim=-1)
    giou_loss = 1.0 - giou_aligned(pred_xyxy, target_xyxy)
    loss = l1_weight * l1 + giou_weight * giou_loss
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def objectness_loss(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, valid.float())
