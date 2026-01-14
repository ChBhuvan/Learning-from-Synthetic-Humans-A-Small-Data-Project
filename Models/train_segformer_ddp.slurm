#!/bin/bash
#SBATCH --job-name=segformer_surreal_ddp
#SBATCH --output=logs/segformer_ddp_%j.out
#SBATCH --error=logs/segformer_ddp_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# ====== env (edit to your conda path) ======
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh # <<EDIT>>
conda activate surreal_arm # <<EDIT>>
# ====== recommended NCCL/runtime settings ======
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 # often safer on mixed clusters; enable IB if you have it
export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# ====== master addr/port for torchrun ======
export MASTER_ADDR=$(hostname)
export MASTER_PORT=23456
# ====== data/output dirs ======
DATA_ROOT= /home/channagiri.b/SmallData_Project/Dataset # <<EDIT>>
OUT_DIR= /home/channagiri.b/SmallData_Project/Output # <<EDIT>>
mkdir -p "$OUT_DIR" logs
# ====== launch (4 GPUs) ======
torchrun --nproc_per_node=4 train_segformer_ddp.py \
--data_root "$DATA_ROOT" \
--output_dir "$OUT_DIR" \
--epochs 30 \
--batch_size 8 \
--num_workers 12 \
--lr 3e-4 \
--weight_decay 1e-4 \
--img_size 320 320 \
--save_every 1 \
--val_every 1 \
--resume '' # set to checkpoint path to resume