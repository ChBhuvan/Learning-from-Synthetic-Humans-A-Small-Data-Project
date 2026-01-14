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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from transformers import SegformerForSemanticSegmentation


# ==========================
# Config
# ==========================
class Config:
    # Paths
    SURREAL_ROOT = "/Users/vaibhavkejriwal/desktop/dataset_surreal/dataset/SURREAL/data"  # change if needed

    # Training
    NUM_CLASSES = 25          # max label in your mask is 24, so 25 classes [0..24]
    IMG_SIZE = (320, 320)     # (H, W) - resize for training
    BATCH_SIZE = 2            # increase on GPU cluster (e.g. 8 or 16)
    NUM_EPOCHS = 5            # start small; you can increase later
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4           # set 0 on Mac if dataloader gives issues

    # Dataset subsampling (so you don't always train on all 5.3M frames)
    MAX_TRAIN_SAMPLES = 50000     # None = all; start small on Mac
    MAX_VAL_SAMPLES = 5000        # for validation speed

    # Checkpoints
    CHECKPOINT_DIR = "./checkpoints_segformer"
    LOG_INTERVAL = 50             # batches


cfg = Config()


# ==========================
# Util: Device selection
# ==========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon M1/M2/M3
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================
# Dataset
# ==========================
class SurrealSegmentationDataset(Dataset):
    """
    Loads SURREAL videos + segmentation .mat files.

    Directory structure assumed:
      SURREAL_ROOT/
        cmu/
          train/
            run0/
              clip1/
                clip1_c0001.mp4
                clip1_c0001_segm.mat
              ...
            run1/...
          val/
            run0/...

    Each *_segm.mat contains keys: segm_1, segm_2, ..., segm_100
    corresponding to each frame in the video.
    """

    def __init__(self, root_dir, split="train",
                 img_size=(320, 320),
                 max_samples=None,
                 augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.base_path = os.path.join(root_dir, "cmu", split)
        self.samples = []  # (video_path, segm_path, frame_idx)

        print(f"[SURREAL] Scanning: {self.base_path}")

        # Recursively walk and index all mp4 + segm.mat pairs
        for root, dirs, files in os.walk(self.base_path):
            mp4_files = [f for f in files if f.endswith(".mp4")]
            for f in mp4_files:
                video_path = os.path.join(root, f)
                segm_path = video_path.replace(".mp4", "_segm.mat")
                if not os.path.exists(segm_path):
                    continue

                # Get number of frames in video
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # SURREAL segmentation .mat stores one frame per key:
                # segm_1, segm_2, ..., so frame_count should align
                for frame_idx in range(frame_count):
                    self.samples.append((video_path, segm_path, frame_idx))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break

                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        print(f"[SURREAL] [{split}] Total samples indexed: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_frame_and_mask(self, video_path, segm_path, frame_idx):
        # ----- Load RGB frame -----
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # H,W,3, uint8

        # ----- Load segmentation mask -----
        mat = loadmat(segm_path)
        seg_key = f"segm_{frame_idx+1}"   # MATLAB is 1-indexed
        if seg_key not in mat:
            # Safety: sometimes keys may be 1..n, but frame_count smaller
            # You can adjust fallback logic here if necessary.
            raise KeyError(f"{seg_key} not found in {segm_path}")
        seg = mat[seg_key]               # H, W

        return frame, seg

    def __getitem__(self, idx):
        video_path, segm_path, frame_idx = self.samples[idx]

        frame_np, seg_np = self._load_frame_and_mask(
            video_path, segm_path, frame_idx
        )

        # Convert to torch tensors
        frame = torch.from_numpy(frame_np).permute(2, 0, 1).float()  # 3,H,W
        seg = torch.from_numpy(seg_np).long()                        # H,W

        # ----- Optional augmentations -----
        # We apply the same random flip to image and mask
        if self.augment:
            if torch.rand(1).item() < 0.5:
                frame = torch.flip(frame, dims=[2])  # horizontal flip (W)
                seg = torch.flip(seg, dims=[1])

        # ----- Resize both image and mask -----
        # Image: bilinear, Mask: nearest
        frame = TF.resize(frame, self.img_size, InterpolationMode.BILINEAR)
        seg = TF.resize(seg.unsqueeze(0).float(),
                        self.img_size,
                        InterpolationMode.NEAREST).squeeze(0).long()

        # Normalize image to [0,1]
        frame = frame / 255.0

        return frame, seg


# ==========================
# Training / Evaluation
# ==========================
def create_dataloaders():
    train_ds = SurrealSegmentationDataset(
        cfg.SURREAL_ROOT,
        split="train",
        img_size=cfg.IMG_SIZE,
        max_samples=cfg.MAX_TRAIN_SAMPLES,
        augment=True,
    )

    val_ds = SurrealSegmentationDataset(
        cfg.SURREAL_ROOT,
        split="val",
        img_size=cfg.IMG_SIZE,
        max_samples=cfg.MAX_VAL_SAMPLES,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def create_model():
    # Basic label mapping [0..24]
    id2label = {i: f"class_{i}" for i in range(cfg.NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}

    # Load pre-trained SegFormer MiT-B2 and adapt to 25 classes
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=cfg.NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # allows us to change num_labels
    )

    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()  # we treat label 0 as background but not ignored
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)          # B,3,H,W
        masks = masks.to(device)           # B,H,W

        optimizer.zero_grad()

        outputs = model(pixel_values=images)
        logits = outputs.logits            # B, num_classes, h, w

        # Resize logits to match mask size (H,W)
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % cfg.LOG_INTERVAL == 0:
            avg_loss = running_loss / cfg.LOG_INTERVAL
            print(f"[Epoch {epoch}] Step {batch_idx+1}/{len(loader)} "
                  f"Loss: {avg_loss:.4f}")
            running_loss = 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)  # B,H,W
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

    avg_loss = total_loss / len(loader.dataset)
    pix_acc = correct_pixels / total_pixels
    return avg_loss, pix_acc


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)
    print(f"[Checkpoint] Saved to {path}")


def main():
    device = get_device()
    print("[Device]", device)

    train_loader, val_loader = create_dataloaders()
    model = create_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    best_val_loss = math.inf

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_pix_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"[Epoch {epoch}] "
              f"Val Loss: {val_loss:.4f}  "
              f"Pixel Acc: {val_pix_acc:.4f}  "
              f"Time: {dt/60:.1f} min")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(
                cfg.CHECKPOINT_DIR,
                f"segformer_b2_best.pth"
            )
            save_checkpoint(model, optimizer, epoch, ckpt_path)


if __name__ == "__main__":
    main()

