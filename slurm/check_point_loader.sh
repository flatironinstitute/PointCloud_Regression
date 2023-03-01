#! /bin/bash 
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --constraint=a100
#SBATCH -c 16
#SBATCH --mem=80gb
#SBATCH --time=16:00:00
 
# Loadinf script for the checkpoint
#paths input here should be in the order of: original cloud data/checkpoint for the adj training/
#checkpoint for the chordal training/output path to save figure and disctionary
 
OUTPUT_DIR=/mnt/ceph/users/$USER/pointnet_run/outputs/$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR
 
python -m util.network_from_chk "/mnt/home/clin/ceph/pointnet_run/data/cloud_and_quat.npz" \
"/mnt/home/clin/ceph/pointnet_run/try/lightning_logs/version_2160644/checkpoints/epoch=999-step=113000.ckpt" \
"/mnt/home/clin/ceph/pointnet_run/OUTPUT_DIR/lightning_logs/version_2161319/checkpoints/epoch=999-step=113000.ckpt" \
"/mnt/home/clin/ceph/pointnet_run/inference"