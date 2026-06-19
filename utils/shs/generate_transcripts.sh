#!/bin/bash
#SBATCH -D .
############# Obligatorias #######################
#SBATCH --time=3-00:00:00               # Consultar batchlim para entender los límites de las particiones.
################# HOST ###########################
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=80              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:4
################ Logging #########################
#SBATCH --job-name=generate_transcripts
#SBATCH --verbose
#SBATCH --output=/path/to/project/outputs/ser_2025/logs/test_evaluation/%x_%j.txt


python /path/to/project/utils/generate_transcripts.py \
	'/path/to/project/data/raw_data/MSP-Podcast2/Audios/Audios' \
	'/path/to/project/data/raw_data/MSP-Podcast2/Labels/test1_labels.tsv' \
	'/path/to/project/data/raw_data/MSP-Podcast2/Labels/test2_labels.tsv' \
	'/path/to/project/data/raw_data/MSP-Podcast2/Labels/test3_labels.tsv' \
	--dump_files_folder '/path/to/project/data/raw_data/MSP-Podcast2/generated_transcripts' \
