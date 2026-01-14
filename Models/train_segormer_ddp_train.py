#!/usr/bin/env python3
# ============================================================
# SegFormer on SURREAL Dataset — DDP-ready Training Script
# Author: Adapted for multi-GPU training (torchrun + SLURM)
# ============================================================

import os
import cv2
import math
import time
import random
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from transformers import SegformerForSemanticSegmentation


# ============================================================
# Config
# ============================================================
class Config:
    # Paths
    SURREAL_ROOT = "/home/channagiri.b/SmallData_Project/Dataset"  # <<EDIT if needed>>

    # Training
    NUM_CLASSES = 25
    IMG_SIZE = (320, 320)
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 12

    # Dataset subsampling
    MAX_TRAIN_SAMPLES = 50000
    MAX_VAL_SAMPLES = 5000

    # Checkpoints
    CHECKPOINT_DIR = "./checkpoints_segformer"
    LOG_INTERVAL = 50


cfg = Config()


# ============================================================
# Utility: Device and DDP Setup
# ============================================================
def setup_ddp():
    """Initialize torch.distributed and return local rank + device."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"[Rank {dist.get_rank()}] Initialized DDP on device {device}")
    return local_rank, device


def cleanup_ddp():
    """Graceful DDP shutdown."""
    dist.destroy_process_group()


# ============================================================
# Dataset
# ============================================================
class SurrealSegmentationDataset(Dataset):
    """
    Loads SURREAL videos + segmentation .mat files.

    Expected directory structure:
      SURREAL_ROOT/
        cmu/train/run*/clip*/clipX_c0001.mp4 and clipX_c0001_segm.mat
    """

    def __init__(self, root_dir, split="train", img_size=(320, 320),
                 max_samples=None, augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.base_path = os.path.join(root_dir, "cmu", split)
        self.samples = []

        if dist.get_rank() == 0:
            print(f"[SURREAL] Scanning: {self.base_path}")

        for root, _, files in os.walk(self.base_path):
            mp4_files = [f for f in files if f.endswith(".mp4")]
            for f in mp4_files:
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

        if dist.get_rank() == 0:
            print(f"[SURREAL] [{split}] Total samples indexed: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_frame_and_mask(self, video_path, segm_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mat = loadmat(segm_path)
        seg_key = f"segm_{frame_idx+1}"
        if seg_key not in mat:
            raise KeyError(f"{seg_key} not found in {segm_path}")
        seg = mat[seg_key]

        return frame, seg

    def __getitem__(self, idx):
        video_path, segm_path, frame_idx = self.samples[idx]
        frame_np, seg_np = self._load_frame_and_mask(video_path, segm_path, frame_idx)

        frame = torch.from_numpy(frame_np).permute(2, 0, 1).float()
        seg = torch.from_numpy(seg_np).long()

        if self.augment and torch.rand(1).item() < 0.5:
            frame = torch.flip(frame, dims=[2])
            seg = torch.flip(seg, dims=[1])

        frame = TF.resize(frame, self.img_size, InterpolationMode.BILINEAR)
        seg = TF.resize(seg.unsqueeze(0).float(),
                        self.img_size, InterpolationMode.NEAREST).squeeze(0).long()

        frame = frame / 255.0
        return frame, seg


# ============================================================
# DataLoader Setup
# ============================================================
def create_dataloaders():
    train_ds = SurrealSegmentationDataset(
        cfg.SURREAL_ROOT, split="train", img_size=cfg.IMG_SIZE,
        max_samples=cfg.MAX_TRAIN_SAMPLES, augment=True,
    )
    val_ds = SurrealSegmentationDataset(
        cfg.SURREAL_ROOT, split="val", img_size=cfg.IMG_SIZE,
        max_samples=cfg.MAX_VAL_SAMPLES, augment=False,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, sampler=val_sampler
    )
    return train_loader, val_loader, train_sampler


# ============================================================
# Model
# ============================================================
def create_model():
    id2label = {i: f"class_{i}" for i in range(cfg.NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=cfg.NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % cfg.LOG_INTERVAL == 0 and dist.get_rank() == 0:
            avg_loss = running_loss / cfg.LOG_INTERVAL
            print(f"[Epoch {epoch}] Step {batch_idx+1}/{len(loader)} Loss: {avg_loss:.4f}")
            running_loss = 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_pixels, correct_pixels = 0.0, 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(pixel_values=images)
        logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

    avg_loss = total_loss / len(loader.dataset)
    pix_acc = correct_pixels / total_pixels
    return avg_loss, pix_acc


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)
    print(f"[Checkpoint] Saved to {path}")


# ============================================================
# Main
# ============================================================
def main():
    local_rank, device = setup_ddp()

    train_loader, val_loader, train_sampler = create_dataloaders()
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    best_val_loss = math.inf

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_pix_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        if dist.get_rank() == 0:
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | Pixel Acc: {val_pix_acc:.4f} | Time: {dt/60:.1f} min")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "segformer_b2_best.pth")
                save_checkpoint(model.module, optimizer, epoch, ckpt_path)

    cleanup_ddp()


if __name__ == "__main__":
    main()
