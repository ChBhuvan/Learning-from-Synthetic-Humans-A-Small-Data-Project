#!/bin/bash
#SBATCH --job-name=segformer_sanitycheck
#SBATCH --output=logs/segformer_sanity_%j.out
#SBATCH --error=logs/segformer_sanity_%j.err
#SBATCH --partition=gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
 
# ====== Conda env setup ======
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate surreal_arm
 
# ====== Runtime env ======
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export MASTER_ADDR=$(hostname)
export MASTER_PORT=23456
 
DATA_ROOT=/home/channagiri.b/SmallData_Project/Dataset/SURREAL/data
OUT_DIR=/home/channagiri.b/SmallData_Project/Output_Sanity
 
mkdir -p "${OUT_DIR}" logs
 
# ====== Launch sanity test (1 GPU, 1 epoch) ======
stdbuf -oL -eL torchrun --nproc_per_node=1 train_segformer_v3.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --epochs 6 \
  --batch_size 8 \
  --num_workers 8 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --img_size 160 160 \
  --save_every 1 \
  --val_every 1 \
  --resume "/home/channagiri.b/SmallData_Project/Output_Sanity/segformer_b2_epoch_5.pth" \
  --max_train_samples 700000 \
  --max_val_samples 50000 