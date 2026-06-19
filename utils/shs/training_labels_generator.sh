#!/bin/bash
#SBATCH --output /path/to/logs/sbatch/ser2025/%x_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:0
#SBATCH --job-name=training_labels_generator

srun python '/path/to/project/utils/training_labels_generator.py' \
	'/path/to/datasets/msp_podcast_2025/Labels/labels_consensus.csv' \
	'/path/to/datasets/msp_podcast_2025/Partitions.txt' \
	--dump_files_folder '/path/to/datasets/msp_podcast_2025/custom_data/generated_training_labels'