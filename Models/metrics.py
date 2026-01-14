import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import SegformerForSemanticSegmentation
from surreal_dataset import SurrealSegmentationDataset


def compute_iou(preds, labels, num_classes):
    eps = 1e-6
    ious = []
    for c in range(num_classes):
        p = (preds == c)
        l = (labels == c)
        inter = (p & l).sum().item()
        union = (p | l).sum().item()
        ious.append(np.nan if union == 0 else inter / (union + eps))
    return ious


@torch.no_grad()
def evaluate(checkpoint, data_root, num_classes=25, img_size=(320,320), batch_size=4, num_workers=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (try with img_size; fallback if needed)
    try:
        val_ds = SurrealSegmentationDataset(data_root, split="val", img_size=img_size, max_samples=None, augment=False)
    except TypeError:
        val_ds = SurrealSegmentationDataset(data_root, split="val", max_samples=None, augment=False)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_ious = []

    ce = torch.nn.CrossEntropyLoss()

    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = model(pixel_values=images).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = ce(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == masks).sum().item()
        total   += masks.numel()

        for i in range(images.size(0)):
            ious = compute_iou(preds[i].cpu(), masks[i].cpu(), num_classes)
            all_ious.append(ious)

    pixel_acc = correct / max(1, total)
    all_ious = np.array(all_ious)  # [N, C]
    per_class_iou = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(per_class_iou)
    avg_loss = total_loss / max(1, len(val_loader.dataset))

    print("\n====== EVAL RESULTS ======")
    print(f"Checkpoint: {checkpoint}")
    print(f"Val Loss : {avg_loss:.4f}")
    print(f"PixelAcc : {pixel_acc:.4f}")
    print(f"mIoU     : {miou:.4f}")
    print("IoU per class:")
    for cid, iou in enumerate(per_class_iou):
        print(f"  class {cid:2d}: {iou:.4f}")
    print("==========================\n")


if __name__ == "__main__":
    # EDIT paths below
    DATA_ROOT = "/path/to/SURREAL/data"                          # <<EDIT>>
    CKPT = "/path/to/output/segformer_surreal_ddp/segformer_b2_best.pth"  # <<EDIT>>
    evaluate(CKPT, DATA_ROOT, num_classes=25, img_size=(320,320))
