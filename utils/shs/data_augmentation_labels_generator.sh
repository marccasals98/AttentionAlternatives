#!/bin/bash
#SBATCH --output /path/to/logs/sbatch/ser2025/%x_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:0
#SBATCH --job-name=data_augmentation_labels_generator

srun python /path/to/project/utils/data_augmentation_labels_generator.py \
	--rirs_data_folders "/path/to/datasets/RIRS_NOISES/real_rirs_isotropic_noises/" "/path/to/datasets/RIRS_NOISES/simulated_rirs/" \
	--noises_data_folders "/path/to/datasets/RIRS_NOISES/pointsource_noises/" "/path/to/datasets/musan/music/" "/path/to/datasets/musan/noise/" "/path/to/datasets/musan/speech" \
	--dump_files_folder '/path/to/datasets/msp_podcast_2025/custom_data/generated_training_labels'