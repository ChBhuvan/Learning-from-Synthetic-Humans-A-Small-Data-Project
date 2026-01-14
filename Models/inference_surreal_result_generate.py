import os
import cv2
import torch
import numpy as np
from scipy.io import loadmat
from torchvision.utils import make_grid
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt


# -------------------------------
# Config
# -------------------------------
DATA_ROOT = "/home/channagiri.b/SmallData_Project/Dataset/SURREAL/data/cmu/val/run0/40_12"
CKPT_PATH = "/home/channagiri.b/SmallData_Project/Output_Sanity/segformer_b2_epoch_6.pth"
OUTPUT_DIR = "/home/channagiri.b/SmallData_Project/Results"
NUM_CLASSES = 25
IMG_SIZE = (160, 160)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Utils
# -------------------------------
def colorize_mask(mask):
    # Simple color map for visualization
    colors = plt.cm.get_cmap('tab20', NUM_CLASSES)
    rgb = colors(mask % NUM_CLASSES)[:, :, :3]
    return (rgb * 255).astype(np.uint8)

def overlay_segmentation(frame, seg_mask, alpha=0.6):
    color_mask = colorize_mask(seg_mask)
    overlay = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return overlay

def load_model(ckpt_path):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Loaded model from {ckpt_path}]")
    return model

# -------------------------------
# Inference on a single video
# -------------------------------
def infer_video(model, video_path, segm_path, out_path, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    print(f"[Processing] {video_path} ({min(frame_count, max_frames)} frames)")
    for idx in range(min(frame_count, max_frames)):
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        inp = F.interpolate(inp, size=IMG_SIZE, mode="bilinear", align_corners=False).to(DEVICE)

        with torch.no_grad():
            logits = model(pixel_values=inp).logits
            logits = F.interpolate(logits, size=(frame_h, frame_w), mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

        overlay = overlay_segmentation(frame_rgb, pred)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        writer.write(overlay_bgr)
    cap.release()
    writer.release()
    print(f"[Saved visualization] {out_path}")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    model = load_model(CKPT_PATH)

    # Example: pick a few random videos from validation set
    for root, _, files in os.walk(DATA_ROOT):
        mp4_files = [f for f in files if f.endswith(".mp4")]
        if not mp4_files: continue
        for f in mp4_files[:10]:  # just 3 for demo
            video_path = os.path.join(root, f)
            segm_path = video_path.replace(".mp4", "_segm.mat")
            name = os.path.splitext(f)[0]
            out_path = os.path.join(OUTPUT_DIR, f"{name}_overlay.mp4")
            infer_video(model, video_path, segm_path, out_path, max_frames=80)
        break
