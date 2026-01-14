import os
import cv2
import torch
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
FREIBURG_IMG_DIR = "/home/channagiri.b/SmallData_Project/Sitting/img"
FREIBURG_MASK_DIR = "/home/channagiri.b/SmallData_Project/Sitting/masks"
CKPT_PATH = "/home/channagiri.b/SmallData_Project/Output_FineTune/segformer_b2_weighted_epoch_7.pth"
OUTPUT_DIR = "/home/channagiri.b/SmallData_Project/Results_Freiburg"
NUM_CLASSES = 25
IMG_SIZE = (160, 160)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Utils
# ============================================================
def colorize_mask(mask):
    """Colorize segmentation mask with distinct colors."""
    cmap = plt.cm.get_cmap("tab20", NUM_CLASSES)
    rgb = cmap(mask % NUM_CLASSES)[:, :, :3]  # RGBA → RGB
    return (rgb * 255).astype(np.uint8)


def overlay_segmentation(frame, seg_mask, alpha=0.6):
    """Overlay colorized segmentation on original frame."""
    color_mask = colorize_mask(seg_mask)
    frame_resized = cv2.resize(frame, (color_mask.shape[1], color_mask.shape[0]))
    overlay = cv2.addWeighted(frame_resized, 1 - alpha, color_mask, alpha, 0)
    return overlay


def load_model(ckpt_path):
    """Load fine-tuned SegFormer model."""
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"[Loaded model from {ckpt_path}]")
    return model


# ============================================================
# Single Image Inference
# ============================================================
@torch.no_grad()
def infer_single(model, img_path, mask_path, out_prefix):
    """Run inference on a single Freiburg image and save visualization."""
    # --- Load RGB image ---
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    inp = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    inp = F.interpolate(inp, size=IMG_SIZE, mode="bilinear", align_corners=False).to(DEVICE)

    # --- Forward pass ---
    logits = model(pixel_values=inp).logits
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    # --- Ground truth mask (.mat) ---
    mat = loadmat(mask_path)
    possible_keys = [k for k in mat.keys() if not k.startswith("__")]
    gt_mask = mat[possible_keys[0]].astype(np.uint8)
    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- Overlays ---
    pred_overlay = overlay_segmentation(img, pred)
    gt_overlay = overlay_segmentation(img, gt_mask)

    # --- Side-by-side comparison ---
    combined = np.concatenate([img, gt_overlay, pred_overlay], axis=1)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_compare.jpg")
    cv2.imwrite(out_path, combined_bgr)
    print(f"[Saved comparison] {out_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    model = load_model(CKPT_PATH)

    # --- Get list of Freiburg images ---
    images = sorted([f for f in os.listdir(FREIBURG_IMG_DIR) if f.endswith(".jpg")])
    if not images:
        raise RuntimeError("No .jpg files found in Freiburg image directory!")

    # --- Select first and last images ---
    first_img = images[0]
    last_img = images[-1]

    # --- Corresponding .mat paths ---
    first_mask = os.path.join(FREIBURG_MASK_DIR, os.path.splitext(first_img)[0] + ".mat")
    last_mask = os.path.join(FREIBURG_MASK_DIR, os.path.splitext(last_img)[0] + ".mat")

    # --- Run inference + save results ---
    print(f"[Visualizing first image: {first_img}]")
    infer_single(model,
                 os.path.join(FREIBURG_IMG_DIR, first_img),
                 first_mask,
                 out_prefix="first")

    print(f"[Visualizing last image: {last_img}]")
    infer_single(model,
                 os.path.join(FREIBURG_IMG_DIR, last_img),
                 last_mask,
                 out_prefix="last")
