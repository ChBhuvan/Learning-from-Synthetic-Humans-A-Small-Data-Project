#!/usr/bin/env python3
import os
import cv2
import math
import time
import random
import argparse
import numpy as np
from typing import Tuple

from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from transformers import SegformerForSemanticSegmentation


# ==========================
# Utilities
# ==========================
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def setup_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_cudnn():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ==========================
# Dataset
# ==========================
class SurrealSegmentationDataset(Dataset):
    """
    Expects:
      <root>/cmu/<split>/.../clipX.mp4
                       clipX_segm.mat  (keys: segm_1, segm_2, ...)
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: Tuple[int, int] = (320, 320),
        max_samples: int | None = None,
        augment: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.base_path = os.path.join(root_dir, "cmu", split)
        self.samples: list[tuple[str, str, int]] = []  # (video_path, segm_path, frame_idx)

        if is_main_process():
            print(f"[SURREAL] Scanning: {self.base_path}")

        # Recursively find mp4 + segm.mat pairs and index per-frame samples
        for root, _dirs, files in os.walk(self.base_path):
            mp4s = [f for f in files if f.endswith(".mp4")]
            for f in mp4s:
                video_path = os.path.join(root, f)
                segm_path = video_path.replace(".mp4", "_segm.mat")
                if not os.path.exists(segm_path):
                    continue

                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                for frame_idx in range(frame_count):
                    self.samples.append((video_path, segm_path, frame_idx))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if is_main_process():
            print(f"[SURREAL] [{split}] Total samples indexed: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_frame_and_mask(self, video_path: str, segm_path: str, frame_idx: int):
        # RGB frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # H,W,3 uint8

        # Segmentation mask
        mat = loadmat(segm_path)
        seg_key = f"segm_{frame_idx + 1}"  # MATLAB is 1-indexed
        if seg_key not in mat:
            raise KeyError(f"{seg_key} not found in {segm_path}")
        seg = mat[seg_key]  # H,W (int labels)

        return frame, seg

    def __getitem__(self, idx: int):
        video_path, segm_path, frame_idx = self.samples[idx]
        frame_np, seg_np = self._load_frame_and_mask(video_path, segm_path, frame_idx)

        frame = torch.from_numpy(frame_np).permute(2, 0, 1).float()  # 3,H,W
        seg = torch.from_numpy(seg_np).long()                        # H,W

        # Simple horizontal flip
        if self.augment:
            if torch.rand(1).item() < 0.5:
                frame = torch.flip(frame, dims=[2])  # width
                seg = torch.flip(seg, dims=[1])

        # Resize (image: bilinear, mask: nearest)
        frame = TF.resize(frame, self.img_size, InterpolationMode.BILINEAR)
        seg = TF.resize(seg.unsqueeze(0).float(), self.img_size, InterpolationMode.NEAREST).squeeze(0).long()

        # Normalize [0,1]
        frame = frame / 255.0
        return frame, seg


# ==========================
# Model / Train / Eval
# ==========================
def create_model(num_classes: int):
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def create_dataloaders(
    data_root: str,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
):
    train_ds = SurrealSegmentationDataset(
        data_root, split="train", img_size=img_size, max_samples=max_train_samples, augment=True
    )
    val_ds = SurrealSegmentationDataset(
        data_root, split="val", img_size=img_size, max_samples=max_val_samples, augment=False
    )

    # DDP requires DistributedSampler
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist_avail_and_initialized() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist_avail_and_initialized() else None

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def train_one_epoch(model, loader, optimizer, device, epoch, log_interval: int):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running = 0.0

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits  # B,C,h,w
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running += loss.item()

        if is_main_process() and (step % log_interval == 0):
            print(f"[Epoch {epoch}] Step {step}/{len(loader)} Loss: {running / log_interval:.4f}")
            running = 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

    avg_loss = total_loss / len(loader.dataset)
    pix_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    return avg_loss, pix_acc


def save_checkpoint(model_or_module, optimizer, epoch: int, path: str, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model_or_module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    if is_main_process():
        print(f"[Checkpoint] Saved to {path}")


def load_checkpoint_if_any(model_or_module, optimizer, resume_path: str) -> int:
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model_or_module.load_state_dict(ckpt["model_state"], strict=False)
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if is_main_process():
            print(f"[Checkpoint] Resumed from {resume_path} (start_epoch={start_epoch})")
        return start_epoch
    return 1


# ==========================
# Main
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="DDP training for SegFormer on SURREAL")
    # data / io
    p.add_argument("--data_root", type=str, required=True, help="SURREAL root (contains cmu/train, cmu/val)")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints/logs")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    # train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, nargs=2, default=[320, 320], metavar=("H", "W"))
    p.add_argument("--num_classes", type=int, default=25)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    # logging / frequency
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--val_every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # DDP init (works for single-GPU too)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = 0  # non-DDP fallback

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if is_main_process():
        print(f"[DDP] world_size={dist.get_world_size() if is_dist_avail_and_initialized() else 1} "
              f"backend={backend} device={device}")

    setup_seed(1337 + get_rank())
    setup_cudnn()

    # Data
    train_loader, val_loader, train_sampler = create_dataloaders(
        data_root=args.data_root,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    # Model / Optimizer
    model = create_model(num_classes=args.num_classes).to(device)
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    output_device=local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    start_epoch = load_checkpoint_if_any(model.module if isinstance(model, DDP) else model,
                                         optimizer, args.resume)

    best_val_loss = math.inf

    # Train loop
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_one_epoch(model, train_loader, optimizer, device, epoch, args.log_interval)

        if (epoch % args.val_every) == 0:
            val_loss, val_pix_acc = evaluate(model, val_loader, device)
            if is_main_process():
                dt = time.time() - t0
                print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}  Pixel Acc: {val_pix_acc:.4f}  Time: {dt/60:.1f} min")

            # Save best (rank 0 only)
            if val_loss < best_val_loss and is_main_process():
                best_val_loss = val_loss
                ckpt_path = os.path.join(args.output_dir, "segformer_b2_best.pth")
                save_checkpoint(model.module if isinstance(model, DDP) else model,
                                optimizer, epoch, ckpt_path,
                                extra={"args": vars(args), "best_val_loss": best_val_loss})

        # Periodic save
        if (epoch % args.save_every) == 0 and is_main_process():
            ckpt_path = os.path.join(args.output_dir, f"segformer_b2_epoch_{epoch}.pth")
            save_checkpoint(model.module if isinstance(model, DDP) else model,
                            optimizer, epoch, ckpt_path, extra={"args": vars(args)})

    # Clean up
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
