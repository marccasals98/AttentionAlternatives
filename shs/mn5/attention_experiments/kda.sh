#!/bin/bash
#SBATCH -D .
############# Obligatorias #######################
#SBATCH --time=2-00:00:00               # Consultar batchlim para entender los límites de las particiones. 
################# HOST ###########################
#SBATCH --gres=gpu:4      # 4 GPUs per node
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
################ Logging #########################
#SBATCH --job-name=ser_2025
#SBATCH --verbose
#SBATCH --output=/path/to/project/outputs/ser_2025/logs/%x_%j.txt



date +%Y-%m-%d_%H:%M:%S

# activate env
source .venv/bin/activate

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


# Use torchrun with uv for distributed data parallel training
# --nproc_per_node should match the number of GPUs requested (#SBATCH --gres=gpu:2)
uv run torchrun --nproc_per_node=4 scripts/train.py \
	--train_data_dir '/path/to/project/data/msp_podcast/Audios' \
	--validation_data_dir '/path/to/project/data/msp_podcast/Audios' \
	--train_labels_path '/path/to/project/data/msp_podcast/custom_data/generated_training_labels/25_01_02_17_03_37_111942/training_labels.tsv' \
	--validation_labels_path '/path/to/project/data/msp_podcast/custom_data/generated_training_labels/25_01_02_17_03_37_111942/development_labels.tsv' \
	--dataset_transcriptions_dir '/path/to/project/data/msp_podcast/custom_data/generated_transcripts/v1/25_01_04_13_38_12_603360/transcripts' \
	--augmentation_noises_labels_path "" \
	--augmentation_rirs_labels_path "" \
	--model_output_folder "/path/to/project/outputs/ser_2025/models" \
	--log_file_folder "/path/to/project/outputs/ser_2025/logs/train" \
	--wandb_dir "/path/to/project/outputs/ser_2025/wandb" \
	--training_random_crop_secs 5.5 \
	--evaluation_random_crop_secs 0 \
	--augmentation_window_size_secs 5.5 \
	--training_augmentation_prob 0 \
	--evaluation_augmentation_prob 0 \
	--augmentation_effects 'apply_speed_perturbation' 'apply_reverb' 'add_background_noise' \
	--speech_feature_extractor 'WAVLM_LARGE' \
	--speech_feature_extractor_output_vectors_dimension 1024 \
	--text_feature_extractor 'BERT_LARGE_UNCASED' \
	--text_feature_extractor_output_vectors_dimension 1024 \
	--speech_adapter 'NoneAdapter' \
	--text_adapter 'NoneAdapter' \
	--seq_to_seq_method 'KDA' \
	--seq_to_seq_heads_number 4 \
	--seq_to_seq_input_dropout 0.4 \
	--seq_to_one_method 'AttentionPooling' \
	--seq_to_one_input_dropout 0.0 \
	--max_epochs 20 \
	--training_batch_size 32 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 2000 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 4 \
	--padding_type 'repetition_pad' \
	--classifier_hidden_layers 4 \
	--classifier_hidden_layers_width 512 \
	--classifier_layer_drop_out 0.1 \
	--number_classes 8 \
	--loss 'CrossEntropy' \
	--weighted_loss \
	--optimizer 'adamw' \
	--update_optimizer_every 2 \
	--learning_rate 0.0001 \
	--learning_rate_multiplier 0.5 \
	--weight_decay 0.01 \
	--use_weights_and_biases \
	--random_seed 1069



date +%Y-%m-%d_%H:%M:%S
