#! /bin/bash 
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --constraint=a100
#SBATCH -c 8
#SBATCH --mem=80gb
#SBATCH --time=16:00:00

#script for the simulator of point clouds
 
OUTPUT_DIR=/mnt/ceph/users/$USER/pointnet_run/data/$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR
 
python -m simulator.simulator output_path="/mnt/home/clin/ceph/pointnet_run/data/cloud_and_quat_max100.npz" max_angle=100 \
hydra.run.dir="$OUTPUT_DIR" batch_size=1000