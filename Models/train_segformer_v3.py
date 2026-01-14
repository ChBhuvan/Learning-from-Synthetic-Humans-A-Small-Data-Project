import os
import cv2
import math
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode
from scipy.io import loadmat
from collections import deque
from transformers import SegformerForSemanticSegmentation

# ============================================================
# Config
# ============================================================
class Config:
    DEFAULT_SURREAL_ROOT = "/home/channagiri.b/SmallData_Project/Dataset/SURREAL/data"
    DEFAULT_OUTPUT_DIR = "/home/channagiri.b/SmallData_Project/Output"
    NUM_CLASSES = 25
    LOG_INTERVAL = 50


cfg = Config()


# ============================================================
# Utils
# ============================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_iou(preds, masks, num_classes):
    """Compute mean Intersection over Union (mIoU)."""
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        mask_c = (masks == c)
        inter = (pred_c & mask_c).sum().item()
        union = (pred_c | mask_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return (sum(ious) / len(ious)) if ious else 0.0


# ============================================================
# Dataset
# ============================================================
class SurrealSegmentationDataset(Dataset):
    """Loads SURREAL videos + segmentation .mat files."""

    def __init__(self, root_dir, split="train", img_size=(320, 320),
                 max_samples=None, augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.base_path = os.path.join(root_dir, "cmu", split)
        self.samples = []

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
                    if max_samples and len(self.samples) >= max_samples:
                        break
                if max_samples and len(self.samples) >= max_samples:
                    break
            if max_samples and len(self.samples) >= max_samples:
                break

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
        seg_key = f"segm_{frame_idx + 1}"
        if seg_key not in mat:
            raise KeyError(f"{seg_key} not found in {segm_path}")
        seg = mat[seg_key]
        return frame, seg

    def __getitem__(self, idx):
        video_path, segm_path, frame_idx = self.samples[idx]
        frame_np, seg_np = self._load_frame_and_mask(video_path, segm_path)
        frame = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
        seg = torch.from_numpy(seg_np).long()
        if self.augment and torch.rand(1).item() < 0.5:
            frame = torch.flip(frame, dims=[2])
            seg = torch.flip(seg, dims=[1])
        frame = F.interpolate(frame.unsqueeze(0), size=self.img_size,
                              mode="bilinear", align_corners=False).squeeze(0)
        seg = F.interpolate(seg.unsqueeze(0).unsqueeze(0).float(),
                            size=self.img_size, mode="nearest").squeeze().long()
        return frame, seg


# ============================================================
# Model + Dataloaders
# ============================================================
def create_model(num_classes):
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


def create_dataloaders(data_root, img_size, batch_size, num_workers,
                       max_train_samples=None, max_val_samples=None):
    train_ds = SurrealSegmentationDataset(data_root, "train",
                                          img_size=img_size,
                                          max_samples=max_train_samples,
                                          augment=True)
    val_ds = SurrealSegmentationDataset(data_root, "val",
                                        img_size=img_size,
                                        max_samples=max_val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ============================================================
# Training and Evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, device, epoch, writer=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running = 0.0
    tic = time.time()
    speeds = deque(maxlen=50)
    total_steps = len(loader)

    for step, (images, masks) in enumerate(loader, start=1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(pixel_values=images)
        logits = F.interpolate(out.logits, size=masks.shape[-2:],
                               mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running += loss.item()
        speeds.append(time.time() - tic)
        tic = time.time()

        if step % 50 == 0 or step == total_steps:
            sec = sum(speeds) / len(speeds)
            remain = total_steps - step
            eta = int(remain * sec)
            avg_loss = running / min(50, step)
            print(f"[Epoch {epoch}] Step {step}/{total_steps} "
                  f"Loss: {avg_loss:.4f} | {sec:.2f}s/step | ETA {eta//3600:02d}:{(eta%3600)//60:02d}m")
            if writer:
                writer.add_scalar("train/loss", avg_loss, (epoch - 1) * total_steps + step)
            running = 0.0


@torch.no_grad()
def evaluate(model, loader, device, writer=None, epoch=0):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    tot_loss, tot_pix, correct_pix, tot_miou, batches = 0, 0, 0, 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        out = model(pixel_values=images)
        logits = F.interpolate(out.logits, size=masks.shape[-2:],
                               mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        preds = logits.argmax(1)
        tot_loss += loss.item() * images.size(0)
        correct_pix += (preds == masks).sum().item()
        tot_pix += masks.numel()
        tot_miou += mean_iou(preds, masks, cfg.NUM_CLASSES)
        batches += 1

    val_loss = tot_loss / len(loader.dataset)
    pix_acc = correct_pix / tot_pix
    miou = tot_miou / max(1, batches)
    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/pixel_acc", pix_acc, epoch)
        writer.add_scalar("val/mIoU", miou, epoch)
    return val_loss, pix_acc, miou


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


def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print(f"[Resume] No checkpoint found at {path}")
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[Resume] Loaded from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=cfg.DEFAULT_SURREAL_ROOT)
    parser.add_argument("--output_dir", type=str, default=cfg.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, nargs=2, default=[320, 320])
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    device = get_device()
    print(f"[Device] Using {device}")

    train_loader, val_loader = create_dataloaders(args.data_root, tuple(args.img_size),
                                                  args.batch_size, args.num_workers)

    model = create_model(cfg.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device) + 1

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
    best_val, patience, bad = math.inf, 5, 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_one_epoch(model, train_loader, optimizer, device, epoch, writer)

        if epoch % args.val_every == 0:
            val_loss, val_pix, val_miou = evaluate(model, val_loader, device, writer, epoch)
            scheduler.step(val_loss)
            dt = (time.time() - t0) / 60
            print(f"[Epoch {epoch}] ValLoss: {val_loss:.4f} | Acc: {val_pix:.4f} | mIoU: {val_miou:.4f} | {dt:.1f} min")

            if val_loss < best_val:
                best_val, bad = val_loss, 0
                save_checkpoint(model, optimizer, epoch, os.path.join(args.output_dir, "segformer_b2_best.pth"))
            else:
                bad += 1
                if bad >= patience:
                    print(f"[EarlyStop] No val improvement for {patience} epochs.")
                    break

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(args.output_dir, f"segformer_b2_epoch_{epoch}.pth"))

    writer.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print("[Training Complete]")


if __name__ == "__main__":
    main()
