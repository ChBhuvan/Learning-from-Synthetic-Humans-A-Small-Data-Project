import os
import glob
from scipy.io import loadmat
import cv2
import torch
from torch.utils.data import Dataset

class SurrealSegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        
        # correct base path
        self.base = os.path.join(root_dir, "cmu", split)

        print("[SURREAL] Scanning:", self.base)

        self.samples = []

        # recursively scan all subdirectories (run0/run1/run2/<clips>/files)
        for root, dirs, files in os.walk(self.base):
            for f in files:
                if f.endswith(".mp4"):
                    video_path = os.path.join(root, f)
                    segm_path = video_path.replace(".mp4", "_segm.mat")
                    if os.path.exists(segm_path):
                        # read number of frames
                        cap = cv2.VideoCapture(video_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        for idx in range(frame_count):
                            self.samples.append((video_path, segm_path, idx))

        print(f"[SURREAL] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, segm_path, frame_idx = self.samples[idx]

        # --- Load the video frame ---
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Load the segmentation mask ---
        mat = loadmat(segm_path)
        seg_key = f"segm_{frame_idx+1}"        # MATLAB-style indexing
        seg = mat[seg_key]                     # (H, W) array

        # --- Convert to tensors ---
        frame = torch.tensor(frame).permute(2, 0, 1).float()
        seg = torch.tensor(seg).long()

        return frame, seg

