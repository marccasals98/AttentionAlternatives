"""
Script for performing inference using a trained model.

Author: Marc Casals Salvador
Date: January 2026
BSC-CNS

"""

import logging
import torch
import os
from settings import LABELS_TO_IDS
import argparse
import time
import numpy as np
from sklearn.metrics import f1_score


from model import Classifier
from data import TrainDataset
from torch.utils.data import DataLoader
from utils import format_training_labels, pad_collate
# region Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
# endregion


# region Configuration
def is_main_process():
    """Check if current process is the main process (rank 0).
    
    Returns:
        bool: True if main process or not using distributed training, False otherwise
    """
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def set_device():
    """Set the device to run the model.
    
    Checks if CUDA is available and sets the device accordingly.
    Logs GPU information if available.
    
    Returns:
        torch.device: Device object (cuda or cpu)
    """
    # Initialize distributed training if using torchrun
    if 'RANK' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Only log on main process
    if is_main_process():
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            gpus_count = torch.cuda.device_count()
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                logger.info(f"Distributed training with {world_size} GPUs")
            else:
                logger.info(f"{gpus_count} GPUs available")
        else:
            logger.info("Using CPU")
    
    return device


# endregion


# region Data loading
def load_data():
    """Load and prepare label mappings.
    
    Creates bidirectional mappings between label IDs and label names.
    Prints the available label names for reference.
    
    Note: This function currently only prints labels and doesn't return values.
    """
    labels_ids = list(range(len(LABELS_TO_IDS)))
    ids_to_labels = {value: key for key, value in LABELS_TO_IDS.items()}

    labels_to_ids = {key: value for key, value in LABELS_TO_IDS.items()}
    labels_names = [ids_to_labels[class_id] for class_id in range(len(LABELS_TO_IDS))]

    print(f"labels_names: {labels_names}")

def set_validation_lines(params):
    """Format and prepare validation labels from file.
    
    Reads the validation labels file and formats it for use with the dataset.
    Prepends the audio data directory to file paths and maps labels to IDs.
    
    Args:
        params: Parameter object containing:
            - validation_labels_path: Path to the validation labels file
            - audios_data_dir: Directory containing audio files
    
    Returns:
        list: Formatted validation labels lines ready for dataset creation
    """
    validation_labels_lines = format_training_labels(
        labels_path = params.validation_labels_path,
        labels_to_ids = LABELS_TO_IDS,
        prepend_directory = params.audios_data_dir,
        header = True,
    )
    return validation_labels_lines

def set_training_lines(params):
    """Format and prepare training labels from file.
    
    Mirrors the validation helper but targets the training labels path so we
    can run inference over the training split only.
    """
    # Support checkpoints that may store this setting as train_labels_path
    training_labels_path = getattr(params, "training_labels_path", None) or getattr(params, "train_labels_path", None)
    training_labels_lines = format_training_labels(
        labels_path = training_labels_path,
        labels_to_ids = LABELS_TO_IDS,
        prepend_directory = params.audios_data_dir,
        header = True,
    )
    return training_labels_lines

def load_validation_data(params, random_crop_secs):
    """Create validation data loader for inference.
    
    Instantiates the validation dataset and creates a DataLoader with appropriate
    parameters. Handles both text-augmented and audio-only models.
    
    Args:
        params: Parameter object containing model configuration:
            - text_feature_extractor: Name of text feature extractor (or 'NoneTextExtractor')
            - evaluation_batch_size: Batch size for evaluation
            - num_workers: Number of data loading workers
            - sample_rate: Audio sample rate
    
    Returns:
        DataLoader: PyTorch DataLoader for validation data
    """
    # Instanciate a Dataset class
    validation_labels_lines = set_validation_lines(params)
    validation_dataset = TrainDataset(
        labels_lines = validation_labels_lines, 
        input_parameters = params,
        random_crop_secs = random_crop_secs,
        augmentation_prob = 0,
    )

    if params.text_feature_extractor != 'NoneTextExtractor':
        data_loader_parameters = {
            'batch_size': params.evaluation_batch_size, 
            'shuffle': False,
            'num_workers': params.num_workers,
            'collate_fn': pad_collate,
            }
    else:
        data_loader_parameters = {
            'batch_size': params.evaluation_batch_size, 
            'shuffle': False,
            'num_workers': params.num_workers,
            }
        
    validation_generator = DataLoader(
        validation_dataset, 
        **data_loader_parameters,
        )
    return validation_generator


def labels_file_has_labels(labels_path):
    """Return True when a TSV labels file has at least filename and label columns."""
    with open(labels_path, "r") as labels_file:
        first_line = labels_file.readline().strip()

    return len(first_line.split("\t")) >= 2

def load_training_data(params):
    """Create training data loader for inference.
    
    Duplicates the validation loader setup but consumes the training labels so
    we can benchmark inference on the training split.
    """
    training_labels_lines = set_training_lines(params)
    training_dataset = TrainDataset(
        labels_lines = training_labels_lines, 
        input_parameters = params,
        random_crop_secs = 512,
        augmentation_prob = 0,
    )

    if params.text_feature_extractor != 'NoneTextExtractor':
        data_loader_parameters = {
            'batch_size': params.evaluation_batch_size, 
            'shuffle': False,
            'num_workers': params.num_workers,
            'collate_fn': pad_collate,
            }
    else:
        data_loader_parameters = {
            'batch_size': params.evaluation_batch_size, 
            'shuffle': False,
            'num_workers': params.num_workers,
            }
        
    training_generator = DataLoader(
        training_dataset, 
        **data_loader_parameters,
        )
    return training_generator
# endregion


# region models
def load_model(device, checkpoint_path):
    """Load trained model from checkpoint.
    
    Loads a pre-trained Classifier model from a checkpoint file. Handles both
    absolute and relative checkpoint paths. For relative paths, automatically
    prepends the default models directory.
    
    Args:
        device (torch.device): Device to load the model onto (cuda or cpu)
        checkpoint_path (str): Path to checkpoint file (absolute or relative)
    
    Returns:
        tuple: (net, params, checkpoint) - Loaded model, parameters, and checkpoint dict
    
    Note:
        Uses weights_only=False for torch.load to support argparse.Namespace objects.
    """
    import os
    
    # If checkpoint_path is just a name, construct full path to models directory
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(
            "/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/models",
            checkpoint_path
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = checkpoint["settings"]

    net = Classifier(params, device)
    net.load_state_dict(checkpoint["model"])
    net.to(device)
    
    return net, params, checkpoint


# endregion

#region evaluate train
def evaluate_training_set(net, device, training_generator, params):
    """Evaluate model on training set.
    
    Runs the model on the training data generator and collects predictions
    and labels. Handles both text-augmented and audio-only models.
    
    Args:
        net (Classifier): Trained model
        device (torch.device): Device to run evaluation on
        training_generator (DataLoader): DataLoader for training data
        params: Parameter object with model configuration:
            - text_feature_extractor: Name of text feature extractor
    
    Note:
        This function currently doesn't return values but collects predictions
        and labels internally.
    """

    with torch.no_grad():
        
        # Switch torch to evaluation mode
        net.eval()
        
        training_final_predictions, training_final_labels = torch.tensor([]).to("cpu"), torch.tensor([]).to("cpu")
        for batch_number, batch_data in enumerate(training_generator):

            if batch_number % 1000 == 0:
                logger.info(f"Evaluating training task batch {batch_number} of {len(training_generator)}...")
            
            if params.text_feature_extractor != 'NoneTextExtractor':
                input, label, transcription_tokens_padded, transcription_tokens_mask = batch_data      
            else:
                input, label = batch_data

            # Assign batch data to device
            if params.text_feature_extractor != 'NoneTextExtractor':
                transcription_tokens_padded, transcription_tokens_mask = transcription_tokens_padded.long().to(device), transcription_tokens_mask.long().to(device)
            input, label = input.float().to(device), label.long().to(device)

            if batch_number == 0: logger.info(f"input.size(): {input.size()}")

            # Calculate prediction and loss
            if params.text_feature_extractor != 'NoneTextExtractor':
                logger.debug("RIGHT")
                prediction  = net.predict(
                    input_tensor = input, 
                    transcription_tokens_padded = transcription_tokens_padded,
                    transcription_tokens_mask = transcription_tokens_mask,
                    )
            else:
                logger.error("WRONG")
                prediction  = net.predict(input_tensor = input)
                #prediction  = net(input_tensor = input)
            #prediction = torch.tensor([prediction]).int()
            #prediction = prediction.to("cpu")
            label = label.to("cpu")

            training_final_predictions = torch.cat(tensors = (training_final_predictions, prediction))
            training_final_labels = torch.cat(tensors = (training_final_labels, label))
#endregion

def bootstrap_macro_f_score_std(labels, predictions, n_bootstrap_samples=1000, random_seed=1234):
    """Estimate macro F-score std by bootstrapping evaluated examples."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    rng = np.random.default_rng(random_seed)
    bootstrap_scores = []

    for _ in range(n_bootstrap_samples):
        sample_indices = rng.integers(0, len(labels), size=len(labels))
        bootstrap_scores.append(
            f1_score(
                y_true=predictions[sample_indices],
                y_pred=labels[sample_indices],
                average="macro",
            )
        )

    return float(np.std(bootstrap_scores))


def evaluate_macro_f_score(net, device, data_generator, params, dataset_name):
    """Run evaluation and print only macro F-score metrics."""
    if is_main_process():
        logger.info(f"Starting metrics-only evaluation for {dataset_name}...")

    net.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_number, batch_data in enumerate(data_generator):
            if params.text_feature_extractor != 'NoneTextExtractor':
                inp, labels, tok, msk = batch_data
                tok = tok.long().to(device)
                msk = msk.long().to(device)
            else:
                inp, labels = batch_data
                tok = None
                msk = None

            inp = inp.float().to(device)

            if params.text_feature_extractor != 'NoneTextExtractor':
                logits = net(
                    input_tensor=inp,
                    transcription_tokens_padded=tok,
                    transcription_tokens_mask=msk,
                )
            else:
                logits = net(input_tensor=inp)

            all_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().long().tolist())

            if (batch_number + 1) % 10 == 0 and is_main_process():
                logger.info(f"Processed {dataset_name} batch {batch_number + 1}/{len(data_generator)}")

    macro_f_score = f1_score(
        y_true=all_predictions,
        y_pred=all_labels,
        average="macro",
    )
    macro_f_score_std = bootstrap_macro_f_score_std(
        labels=all_labels,
        predictions=all_predictions,
    )

    if is_main_process():
        print("=" * 70)
        print(f"METRICS RESULTS: {dataset_name}")
        print("=" * 70)
        print(f"Total samples processed: {len(all_labels)}")
        print(f"Macro F-score:      {macro_f_score:.3f}")
        print(f"Macro F-score std:  {macro_f_score_std:.3f}")
        print("=" * 70)

    return {
        "dataset_name": dataset_name,
        "total_samples": len(all_labels),
        "macro_f_score": macro_f_score,
        "macro_f_score_std": macro_f_score_std,
    }


def run_inference(net, device, data_generator, params):
    """
    Run inference and measure SEQ2SEQ-ONLY timing + GPU memory.

    - Timing is measured only for the seq2seq layer (CUDA events inside the model).
    - Memory is the peak additional allocated GPU memory during the seq2seq call
      (baseline at seq2seq entry subtracted).
    """

    if is_main_process():
        logger.info("Starting SEQ2SEQ-only inference profiling...")

    net.eval()

    seq2seq_peak_mb_samples = []
    seq2seq_time_ms_per_sample = []
    all_predictions = []
    all_labels = []
    total_samples = 0

    # -------------------------
    # Warm-up (important to avoid measuring one-time allocations / kernel selection)
    # -------------------------
    if is_main_process():
        logger.info("Running warm-up pass (1 batch)...")

    with torch.no_grad():
        for batch_number, batch_data in enumerate(data_generator):
            if batch_number >= 1:
                break

            if params.text_feature_extractor != 'NoneTextExtractor':
                inp, _, tok, msk = batch_data
                tok = tok.long().to(device)
                msk = msk.long().to(device)
            else:
                inp, _ = batch_data
                tok = torch.zeros((inp.size(0), 1), dtype=torch.long, device=device)
                msk = torch.zeros((inp.size(0), 1), dtype=torch.long, device=device)

            inp = inp.float().to(device)

            # Warm up the exact code path we measure
            _ = net.forward_with_seq_to_seq_profiling(
                input_tensor=inp,
                transcription_tokens_padded=tok,
                transcription_tokens_mask=msk,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # -------------------------
    # Timed profiling
    # -------------------------
    if is_main_process():
        logger.info("Starting timed profiling...")

    overall_start_time = time.time()

    with torch.no_grad():
        for batch_number, batch_data in enumerate(data_generator):

            if params.text_feature_extractor != 'NoneTextExtractor':
                inp, labels, tok, msk = batch_data
                tok = tok.long().to(device)
                msk = msk.long().to(device)
            else:
                inp, labels = batch_data
                tok = torch.zeros((inp.size(0), 1), dtype=torch.long, device=device)
                msk = torch.zeros((inp.size(0), 1), dtype=torch.long, device=device)

            batch_size = inp.size(0)
            inp = inp.float().to(device)

            logits, seq2seq_peak_mb, seq2seq_time_ms = net.forward_with_seq_to_seq_profiling(
                input_tensor=inp,
                transcription_tokens_padded=tok,
                transcription_tokens_mask=msk,
            )

            total_samples += batch_size
            all_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().long().tolist())

            # Memory: per-batch peak delta MB (already seq2seq-only)
            if seq2seq_peak_mb is not None:
                seq2seq_peak_mb_samples.append(seq2seq_peak_mb)

            # Time: convert batch time -> per-sample time for generality
            if seq2seq_time_ms is not None:
                seq2seq_time_ms_per_sample.append(seq2seq_time_ms / max(1, batch_size))

            if (batch_number + 1) % 10 == 0 and is_main_process():
                logger.info(f"Processed batch {batch_number + 1}/{len(data_generator)}")

    overall_end_time = time.time()
    wall_time_s = overall_end_time - overall_start_time

    # -------------------------
    # Aggregate stats
    # -------------------------
    seq2seq_time_stats = None
    if seq2seq_time_ms_per_sample:
        mean_ms = sum(seq2seq_time_ms_per_sample) / len(seq2seq_time_ms_per_sample)
        max_ms = max(seq2seq_time_ms_per_sample)
        min_ms = min(seq2seq_time_ms_per_sample)

        # throughput based on seq2seq time only (mean per sample)
        # If batch size = 1, this equals 1000 / mean_ms
        throughput = 1000.0 / mean_ms if mean_ms > 0 else None

        seq2seq_time_stats = {
            "mean_ms_per_sample": mean_ms,
            "max_ms_per_sample": max_ms,
            "min_ms_per_sample": min_ms,
            "throughput_samples_per_s": throughput,
        }

    seq2seq_mem_stats = None
    if seq2seq_peak_mb_samples:
        seq2seq_mem_stats = {
            "mean_peak_mb": sum(seq2seq_peak_mb_samples) / len(seq2seq_peak_mb_samples),
            "max_peak_mb": max(seq2seq_peak_mb_samples),
            "min_peak_mb": min(seq2seq_peak_mb_samples),
        }

    f_score_stats = None
    if all_labels:
        f_score_stats = {
            "macro_f_score": f1_score(
                y_true=all_predictions,
                y_pred=all_labels,
                average="macro",
            ),
            "macro_f_score_std": bootstrap_macro_f_score_std(
                labels=all_labels,
                predictions=all_predictions,
            ),
        }

    # -------------------------
    # Print
    # -------------------------
    if is_main_process():
        print("=" * 70)
        print("SEQ2SEQ-ONLY PROFILING RESULTS")
        print("=" * 70)
        print(f"Total samples processed: {total_samples}")
        print(f"Total wall time (includes dataloader + full forward): {wall_time_s:.4f} s")

        if seq2seq_time_stats is not None:
            print("-" * 70)
            print("SEQ2SEQ TIME")
            print("-" * 70)
            print(f"Mean time per sample: {seq2seq_time_stats['mean_ms_per_sample']:.3f} ms")
            print(f"Max time per sample:  {seq2seq_time_stats['max_ms_per_sample']:.3f} ms")
            print(f"Min time per sample:  {seq2seq_time_stats['min_ms_per_sample']:.3f} ms")
            if seq2seq_time_stats["throughput_samples_per_s"] is not None:
                print(f"Seq2seq throughput:   {seq2seq_time_stats['throughput_samples_per_s']:.2f} samples/s")

        if seq2seq_mem_stats is not None:
            print("-" * 70)
            print("SEQ2SEQ PEAK GPU MEMORY (Peak additional allocated during seq2seq)")
            print("-" * 70)
            print(f"Mean peak: {seq2seq_mem_stats['mean_peak_mb']:.3f} MB ({seq2seq_mem_stats['mean_peak_mb']/1024:.4f} GB)")
            print(f"Max peak:  {seq2seq_mem_stats['max_peak_mb']:.3f} MB ({seq2seq_mem_stats['max_peak_mb']/1024:.4f} GB)")
            print(f"Min peak:  {seq2seq_mem_stats['min_peak_mb']:.3f} MB ({seq2seq_mem_stats['min_peak_mb']/1024:.4f} GB)")

        if f_score_stats is not None:
            print("-" * 70)
            print("CLASSIFICATION METRICS")
            print("-" * 70)
            print(f"Macro F-score:      {f_score_stats['macro_f_score']:.3f}")
            print(f"Macro F-score std:  {f_score_stats['macro_f_score_std']:.3f}")

        print("=" * 70)

    return {
        "total_samples": total_samples,
        "wall_time_s": wall_time_s,
        "seq2seq_time_stats": seq2seq_time_stats,
        "seq2seq_mem_stats": seq2seq_mem_stats,
        "f_score_stats": f_score_stats,
    }

def count_parameters(module):
    """Return total and trainable parameter counts for a module."""
    # Some modules define an attribute named `parameters`, shadowing the nn.Module method;
    # use named_parameters() to avoid collisions with such attributes.
    total = 0
    trainable = 0
    for _, p in module.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def log_model_param_breakdown(net):
    """Log parameter counts for key model components."""
    parts = [
        ("speech_feature_extractor", net.speech_feature_extractor),
        ("text_feature_extractor", net.text_feature_extractor),
        ("speech_adapter_layer", net.speech_adapter_layer),
        ("text_adapter_layer", net.text_adapter_layer),
        ("seq_to_seq_layer", net.seq_to_seq_layer),
        ("seq_to_one_layer", net.seq_to_one_layer),
        ("classifier_layer", net.classifier_layer),
    ]

    total_all, trainable_all = 0, 0
    for name, module in parts:
        total, trainable = count_parameters(module)
        total_all += total
        trainable_all += trainable
        logger.info("Params %s: total=%d, trainable=%d", name, total, trainable)

    logger.info("Params TOTAL: total=%d, trainable=%d", total_all, trainable_all)


def parse_arguments():
    """Parse command-line arguments for inference script.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - checkpoint: Path to model checkpoint file
            - audios_data_dir: Directory containing audio files
            - dataset: Dataset to use ('training' or 'validation')
            - training_labels_path: Path to training labels CSV file
            - validation_labels_path: Path to validation labels CSV file
    """
    parser = argparse.ArgumentParser(description="Inference script for trained model.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--audios-data-dir", type=str, required=True, help="Path to the audio data directory."
    )
    parser.add_argument(
        "--dataset", type=str, choices=["training", "validation"], default="validation",
        help="Dataset to use for inference: 'training' or 'validation'. Default: training."
    )
    parser.add_argument(
        "--training-labels-path", type=str, required=False, help="Path to the training labels CSV file (required if --dataset=training)."
    )
    parser.add_argument(
        "--validation-labels-path", type=str, required=False, help="Path to the validation labels CSV file (required if --dataset=validation)."
    )
    parser.add_argument(
        "--evaluation-batch-size", type=int, required=False, default=None, help="Optional: override evaluation batch size from checkpoint. Default: use checkpoint value."
    )
    parser.add_argument(
        "--random_crop_secs", type=int, required=False, default=300, help="Optional: seconds to randomly crop audio during inference. Default: 300 seconds."
    )
    parser.add_argument(
        "--metrics-only", action="store_true", help="Only compute macro F-score metrics; skip timing and GPU memory profiling."
    )
    parser.add_argument(
        "--dataset-name", type=str, required=False, default=None, help="Name to print for the evaluated split."
    )
    
    args, unknown = parser.parse_known_args()
    
    # Validate that the required labels path is provided based on dataset choice
    if args.dataset == "training" and not args.training_labels_path:
        parser.error("--training-labels-path is required when --dataset=training")
    if args.dataset == "validation" and not args.validation_labels_path:
        parser.error("--validation-labels-path is required when --dataset=validation")
    
    if unknown and is_main_process():
        logger.info(f"Ignoring unknown CLI args: {unknown}")
    return args


def main():
    """Main entry point for inference timing script.
    
    Orchestrates the complete inference pipeline:
    1. Parses command-line arguments
    2. Sets up the device (GPU or CPU)
    3. Loads the trained model from checkpoint
    4. Loads dataset data
    5. Runs inference with timing measurements
    6. Reports timing results
    
    The script measures and logs detailed timing metrics including:
    - Total inference time
    - Average time per batch and per sample
    - Throughput (samples/second)
    """
    args = parse_arguments()
    device = set_device()

    selected_labels_path = args.training_labels_path if args.dataset == "training" else args.validation_labels_path
    dataset_name = args.dataset_name or args.dataset

    if args.metrics_only and selected_labels_path and not labels_file_has_labels(selected_labels_path):
        if is_main_process():
            print("=" * 70)
            print(f"METRICS RESULTS: {dataset_name}")
            print("=" * 70)
            print(f"Cannot compute Macro F-score for {dataset_name}: labels file has no label column.")
            print(f"Labels file: {selected_labels_path}")
            print("=" * 70)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    # print args
    for arg_name, arg_value in vars(args).items():
        logger.info(f"Argument {arg_name}: {arg_value}")
    
    if is_main_process():
        logger.info(f"Device set to: {device}")
        logger.info(f"Selected dataset for inference: {args.dataset}")

    # Load model and get full parameters from checkpoint
    net, params, checkpoint = load_model(device, args.checkpoint)
    if is_main_process():
        logger.info(
            "Checkpoint config: seq_to_seq_method=%s, seq_to_one_method=%s, text_feature_extractor=%s, padding_type=%s",
            getattr(params, "seq_to_seq_method", None),
            getattr(params, "seq_to_one_method", None),
            getattr(params, "text_feature_extractor", None),
            getattr(params, "padding_type", None),
        )
        log_model_param_breakdown(net)

    # Update params with command-line arguments
    params.audios_data_dir = args.audios_data_dir
    
    # Override batch size if provided
    if args.evaluation_batch_size is not None:
        original_batch_size = params.evaluation_batch_size
        params.evaluation_batch_size = args.evaluation_batch_size
        if is_main_process():
            logger.info(f"Overriding evaluation_batch_size from {original_batch_size} to {args.evaluation_batch_size}")
    
    # Load the appropriate dataset based on user selection
    if args.dataset == "training":
        if is_main_process():
            logger.info(f"Loading training dataset from: {args.training_labels_path}")
        # Set both fields for compatibility with legacy checkpoints
        params.training_labels_path = args.training_labels_path
        params.train_labels_path = args.training_labels_path
        data_generator = load_training_data(params)
    else:  # validation
        if is_main_process():
            logger.info(f"Loading validation dataset from: {args.validation_labels_path}")
        params.validation_labels_path = args.validation_labels_path
        data_generator = load_validation_data(params=params, random_crop_secs=args.random_crop_secs)
    
    if is_main_process():
        logger.info(f"Using evaluation_batch_size: {params.evaluation_batch_size}")

    # Run inference and measure timing
    if args.metrics_only:
        evaluate_macro_f_score(net, device, data_generator, params, dataset_name)
    else:
        run_inference(net, device, data_generator, params)
    
    if is_main_process():
        logger.info("Inference timing measurement completed successfully!")
    
    # Clean up distributed process group if initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
