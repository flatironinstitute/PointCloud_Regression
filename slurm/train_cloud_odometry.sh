#! /bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --constraint=a100
#SBATCH -c 16
#SBATCH --mem=80gb
#SBATCH --time=16:00:00

 # Training script for the simulated point cloud
  
 OUTPUT_DIR=/mnt/ceph/users/$USER/pointnet_run/outputs/$SLURM_JOB_ID
 mkdir -p $OUTPUT_DIR
  
 python -m regression.trainer data.file_path="/mnt/home/clin/ceph/pointnet_run/data/cloud_and_quat.npz" \
 batch_size=8 model_config.adj_option=True model_config.batch_norm=True hydra.run.dir='/mnt/home/clin/ceph/pointnet_run/try' num_epochs=1000