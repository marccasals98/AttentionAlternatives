#!/bin/bash
#SBATCH -D .
############# Obligatorias #######################
#SBATCH --time=3-00:00:00               # Consultar batchlim para entender los límites de las particiones. 
################# HOST ###########################
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=80              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:1
################ Logging #########################
#SBATCH --job-name=evaluate_inference
#SBATCH --verbose
#SBATCH --output=/path/to/project/outputs/ser_2025/logs/inference_evaluation/%x_%j.txt

date +%Y-%m-%d_%H:%M:%S

# activate env
source .venv/bin/activate
# source /path/to/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
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

uv run torchrun --nproc_per_node=1 scripts/inference_all_sets.py \
    --checkpoint '/path/to/project/outputs/ser_2025/models/26_01_09_15_32_39_WAV2VEC2_XLSR_300M_BERT_LARGE_UNCASED_NoneAdapter_NoneAdapter_MultiHeadStandardVersion_AttentionPooling_gln059kk/26_01_09_15_32_39_WAV2VEC2_XLSR_300M_BERT_LARGE_UNCASED_NoneAdapter_NoneAdapter_MultiHeadStandardVersion_AttentionPooling_gln059kk.chkpt' \
    --audios-data-dir '/path/to/project/data/msp_podcast/Audios' \
    --dev-labels-path '/path/to/project/data/msp_podcast/custom_data/generated_training_labels/25_01_02_17_03_37_111942/toy_training_labels.tsv' \
    --validation-labels-path '/path/to/project/data/msp_podcast/custom_data/generated_training_labels/25_01_02_17_03_37_111942/toy_development_labels.tsv' \
    --evaluation-batch-size 1 \
    --dataset validation \
    --random_crop_secs 600

date +%Y-%m-%d_%H:%M:%S