# SER2025
This repository is based on an INTERSPEECH Speech Emotion Recognition Challenge, where we placed 7th.

## Using `uv` in this project
This repository uses `uv` in the shell scripts to run commands in an isolated environment.
The provided `.sh` scripts are intended for use on HPC systems with job schedulers and are typically submitted with `sbatch shs/<script>.sh`.
You can also use `uv run` locally for commands outside the HPC submission workflow.

### 1. Install and activate `uv`
If you do not have `uv` installed, follow the `uv` installation instructions for your environment.

### 2. Run training with `uv`
From the repository root, use a command like:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py \
  --train_data_dir /path/to/project/data/msp_podcast/Audios \
  --validation_data_dir /path/to/project/data/msp_podcast/Audios \
  --train_labels_path /path/to/project/data/msp_podcast/custom_data/generated_training_labels/<label-file>.tsv \
  --validation_labels_path /path/to/project/data/msp_podcast/custom_data/generated_training_labels/<label-file>.tsv \
  --dataset_transcriptions_dir /path/to/project/data/msp_podcast/custom_data/generated_transcripts/v1/<transcript-folder>/transcripts \
  --model_output_folder /path/to/project/outputs/ser_2025/models \
  --log_file_folder /path/to/project/outputs/ser_2025/logs/train \
  --wandb_dir /path/to/project/outputs/ser_2025/wandb
```

### 3. Use `uv` for other scripts
For inference or evaluation scripts, `uv run` can wrap the Python command in the same way:

```bash
uv run python scripts/inference.py --checkpoint /path/to/checkpoint.chkpt --audios-data-dir /path/to/audios
```

### 4. Notes
- Replace placeholder paths with your local dataset, model, and output directories.
- The `uv` wrapper ensures command isolation and environment consistency across runs.

## Performance evaluation
To perform a performanance evaluation it is needed to run the following experiment:
`sbatch shs/mn5/inference_eval/evaluate_all_sets.sh`. This inmediately calls `inference_all_sets.py` file.

The `inferencе_all_sets.py` profiling logic reports the following metrics:

- `Total wall time`: the full runtime for the inference evaluation pass, including data loading and the model forward pass.
- `Mean time per sample`: average latency for the seq2seq portion of the model, reported in milliseconds per sample.
- `Max time per sample`: worst-case seq2seq latency observed during the run.
- `Min time per sample`: best-case seq2seq latency observed during the run.
- `Throughput samples/s`: estimated sample throughput computed from mean seq2seq latency.
- `Mean/Max/Min peak GPU memory`: peak additional GPU memory used during the seq2seq layer execution, reported in MB and GB.

The GPU peak memory metric is focused on the seq2seq stage only, not the full process memory footprint. The latency metrics are derived from seq2seq timing and are most meaningful when running with batch size 1, as in the current inference evaluation setup.
