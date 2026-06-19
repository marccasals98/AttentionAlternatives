#!/bin/bash
#SBATCH -D .
############# Obligatorias #######################
#SBATCH --time=0-02:00:00               # Consultar batchlim para entender los límites de las particiones.
################# HOST ###########################
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=80              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:4
################ Logging #########################
#SBATCH --job-name=evaluate_test
#SBATCH --verbose
#SBATCH --output=/path/to/project/outputs/ser_2025/logs/test_evaluation/%x_%j.txt

date +%Y-%m-%d_%H:%M:%S

# activate env
source .venv/bin/activate
# source /path/to/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export WANDB_CACHE_DIR="/path/to/project/outputs/ser_2025/cache/wandb"
export WANDB_CONFIG_DIR="/path/to/project/outputs/ser_2025/cache/wandb/config"
export WANDB_DATA_DIR="/path/to/project/outputs/ser_2025/cache/wandb/data"
export TORCH_HOME="/path/to/project/outputs/ser_2025/cache/torch"
export HUGGINGFACE_HUB_CACHE="/path/to/project/outputs/ser_2025/cache/huggingface"
export HF_HOME="/path/to/project/outputs/ser_2025/cache/huggingface"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1234
export OMP_NUM_THREADS=20

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use torchrun with uv for distributed data parallel training
# --nproc_per_node should match the number of GPUs requested (#SBATCH --gres=gpu:2)

uv run torchrun --nproc_per_node=4 scripts/get_test_scores.py \
    --checkpoint '/path/to/project/outputs/ser_2025/models/26_01_20_20_10_43_WAVLM_LARGE_BERT_LARGE_UNCASED_NoneAdapter_NoneAdapter_GSA_AttentionPooling_zbu20ndk/26_01_20_20_10_43_WAVLM_LARGE_BERT_LARGE_UNCASED_NoneAdapter_NoneAdapter_GSA_AttentionPooling_zbu20ndk.chkpt' \
    --audios-data-dir '/path/to/project/data/raw_data/MSP-Podcast2/Audios/Audios' \
    --test-labels-path '/path/to/project/data/raw_data/MSP-Podcast2/Labels/labels_consensus.csv' \
    --evaluation-batch-size 1 \
    --random_crop_secs None \
    --transcriptions-dir '/path/to/project/data/raw_data/MSP-Podcast2/Transcripts'

date +%Y-%m-%d_%H:%M:%S