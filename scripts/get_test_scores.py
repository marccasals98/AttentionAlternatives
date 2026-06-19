import argparse
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import glob

# Suppress torchaudio deprecation warning about torchcodec
warnings.filterwarnings("ignore", message=".*In 2.9, this function's implementation will be changed.*")

# import inference and data helpers
from inference import set_device, load_model
from settings import LABELS_TO_IDS
from data import TrainDataset
from torch.utils.data import DataLoader
from utils import pad_collate

def resolve_audio_paths(audios_dir: str, fname: str):
    """
    Returns a list of existing wav paths for this FileName.
    Handles both:
      MSP-PODCAST_XXXX_YYYY.wav
      MSP-PODCAST_XXXX_YYYY_0001.wav (and more segments)
    """
    direct = os.path.join(audios_dir, fname)
    if os.path.exists(direct):
        return [direct]

    base = os.path.splitext(os.path.basename(fname))[0]  # MSP-PODCAST_XXXX_YYYY
    # search recursively because you have Audios/Audios/ nesting
    pattern = os.path.join(audios_dir, "**", base + "_*.wav")
    segs = sorted(glob.glob(pattern, recursive=True))
    return segs

def _parse_random_crop(value: str):
	"""Convert a string to float or None for --random_crop_secs."""
	if value is None:
		return None
	v = str(value).strip()
	if v.lower() in ("none", "null", ""):
		return None
	try:
		return float(v)
	except ValueError:
		return None


def compute_bootstrap_ci(y_true, y_pred, n_iterations=1000, ci_level=0.95):
	"""
	Compute bootstrap confidence interval for macro F1 score.

	Args:
		y_true: True labels array
		y_pred: Predicted labels array
		n_iterations: Number of bootstrap samples
		ci_level: Confidence level (e.g., 0.95 for 95% CI)

	Returns:
		dict with 'f1_mean', 'f1_median', 'ci_lower', 'ci_upper'
	"""
	np.random.seed(42)  # for reproducibility
	n_samples = len(y_true)
	f1_scores = []

	for _ in range(n_iterations):
		# Resample with replacement
		indices = np.random.choice(n_samples, size=n_samples, replace=True)
		y_true_boot = np.array(y_true)[indices]
		y_pred_boot = np.array(y_pred)[indices]

		# Compute F1 for this bootstrap sample
		f1_boot = f1_score(y_true_boot, y_pred_boot, average='macro')
		f1_scores.append(f1_boot)

	f1_scores = np.array(f1_scores)
	alpha = 1 - ci_level
	ci_lower = np.percentile(f1_scores, 100 * alpha / 2)
	ci_upper = np.percentile(f1_scores, 100 * (1 - alpha / 2))

	return {
		'f1_mean': float(np.mean(f1_scores)),
		'f1_median': float(np.median(f1_scores)),
		'ci_lower': float(ci_lower),
		'ci_upper': float(ci_upper)
	}


def parse_args():
	parser = argparse.ArgumentParser(description="Compute test scores for MSP-Podcast test sets")
	parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint file")
	parser.add_argument("--audios-data-dir", required=True, help="Directory containing audio files")
	parser.add_argument("--transcriptions-dir", default=None, help="Directory containing transcription .txt files (optionally in a 'transcripts/' subfolder)")
	parser.add_argument("--test-labels-path", required=True, help="Path to labels_consensus.csv")
	parser.add_argument("--evaluation-batch-size", type=int, default=1, help="Batch size for evaluation")
	parser.add_argument("--random_crop_secs", default=None, help="Random crop seconds (float) or 'None'")
	parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node (torchrun)")
	parser.add_argument("--bootstrap-n-iterations", type=int, default=1000, help="Number of bootstrap iterations for confidence intervals")
	parser.add_argument("--bootstrap-ci-level", type=float, default=0.95, help="Confidence level for bootstrap CI (e.g., 0.95 for 95% CI)")
	return parser.parse_args()


def evaluate_split(split_name: str, df: pd.DataFrame, net, params, device, args, labels_map, transcriptions_dir=None):
	df_split = df[df["Split_Set"] == split_name]
	total_expected = len(df_split)
	if total_expected == 0:
		print(f"No rows for split {split_name}, skipping.")
		return None

	# Resolve the effective transcription directory once. TrainDataset expects
	# params.dataset_transcriptions_dir to directly contain *.txt files.
	trans_dir = transcriptions_dir
	if trans_dir is None:
		trans_dir = getattr(params, 'dataset_transcriptions_dir', None)
	if trans_dir is None:
		# if not provided in params, try common alternate names
		trans_dir = getattr(params, 'dataset_transcriptions_path', None)
	if trans_dir is not None and os.path.isdir(os.path.join(trans_dir, 'transcripts')):
		trans_dir = os.path.join(trans_dir, 'transcripts')

	def _normalize_emo(raw):
		if raw is None:
			return None
		s = str(raw).strip()
		if s == "":
			return None
		s_low = s.lower()
		# direct match
		if s_low in labels_map:
			return s_low
		# single-letter uppercase like 'N' -> 'n'
		if len(s_low) >= 1 and s_low[0] in labels_map:
			return s_low[0]
		# common full-word mappings
		word_map = {
			'neutral': 'n', 'happy': 'h', 'angry': 'a', 'sad': 's', 'surprise': 'u', 'disgust': 'd', 'fear': 'f', 'calm': 'c'
		}
		if s_low in word_map:
			return word_map[s_low]
		return None

	labels_lines = []
	missing_files = []
	malformed_transcriptions = []

	for _, row in df_split.iterrows():
		fname = row["FileName"]
		emo = _normalize_emo(row["EmoClass"])

		# validate label
		if emo not in labels_map:
			# bad/unknown label -> skip (don't count as missing file)
			continue
		label_id = labels_map[emo]

		# resolve audio path(s)
		paths = resolve_audio_paths(args.audios_data_dir, fname)
		if not paths:
			missing_files.append(os.path.join(args.audios_data_dir, fname))
			continue

		# for each resolved wav (handles segmented audio)
		for file_path in paths:
			# Optional transcript check only if text features are enabled
			if params.text_feature_extractor != 'NoneTextExtractor' and trans_dir is not None:
				base = os.path.basename(file_path)
				txt_name = os.path.splitext(base)[0] + ".txt"
				transcription_path = os.path.join(trans_dir, txt_name)

				if not os.path.exists(transcription_path):
					missing_files.append(transcription_path)
					continue

				try:
					with open(transcription_path, "r") as f:
						lines = f.readlines()
				except OSError:
					malformed_transcriptions.append(transcription_path)
					missing_files.append(transcription_path)
					continue

				if len(lines) != 1:
					malformed_transcriptions.append(transcription_path)
					missing_files.append(transcription_path)
					continue

			labels_lines.append(f"{file_path}\t{label_id}")

	missing_count = len(missing_files)
	missing_pct = 100.0 * missing_count / total_expected if total_expected > 0 else 0.0
	print(f"Split {split_name}: expected={total_expected}, found={len(labels_lines)}, missing={missing_count} ({missing_pct:.2f}%)")
	if malformed_transcriptions:
		print(f"Split {split_name}: discarded malformed transcriptions={len(malformed_transcriptions)}")

	if len(labels_lines) == 0:
		print(f"No valid files to evaluate for split {split_name}.")
		return {"split": split_name, "n": 0, "f1_macro": None, "missing": missing_count, "malformed_transcriptions": len(malformed_transcriptions)}

	# Update params for this evaluation
	params.audios_data_dir = args.audios_data_dir
	if trans_dir is not None:
		params.dataset_transcriptions_dir = trans_dir
		params.dataset_transcriptions_path = trans_dir
	if args.evaluation_batch_size is not None:
		params.evaluation_batch_size = args.evaluation_batch_size

	# Determine random crop seconds: CLI overrides checkpoint; ensure numeric
	if args.random_crop_secs is not None:
		rc_secs = args.random_crop_secs
	else:
		rc_secs = getattr(params, 'evaluation_random_crop_secs', None) or getattr(params, 'training_random_crop_secs', 2.0)

	# Instantiate dataset and dataloader directly (avoid format_training_labels assertions)
	dataset = TrainDataset(labels_lines=labels_lines,
						input_parameters=params,
						random_crop_secs=rc_secs,
						augmentation_prob=0)

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

	data_generator = DataLoader(dataset, **data_loader_parameters)

	# Run model to collect predictions and labels
	y_true = []
	y_pred = []

	net.eval()
	import torch
	with torch.no_grad():
		for batch in data_generator:
			if params.text_feature_extractor != 'NoneTextExtractor':
				inp, label, tok, msk = batch
				tok = tok.long().to(device)
				msk = msk.long().to(device)
				inp = inp.float().to(device)
				logits = net(inp, transcription_tokens_padded=tok, transcription_tokens_mask=msk)
			else:
				inp, label = batch
				inp = inp.float().to(device)
				logits = net(inp)

			preds = torch.argmax(logits, dim=1).to("cpu").numpy()
			labels = label.to("cpu").numpy()

			y_pred.extend(preds.tolist())
			y_true.extend(labels.tolist())

	# compute macro F1 and bootstrap CI
	f1 = f1_score(y_true, y_pred, average="macro")
	bootstrap_results = compute_bootstrap_ci(
		y_true, y_pred,
		n_iterations=args.bootstrap_n_iterations,
		ci_level=args.bootstrap_ci_level
	)
	print(f"Split {split_name}: samples={len(y_true)}, F1_macro={f1:.4f}, CI=[{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
	return {
		"split": split_name,
		"n": len(y_true),
		"f1_macro": float(f1),
		"f1_mean": bootstrap_results['f1_mean'],
		"f1_median": bootstrap_results['f1_median'],
		"ci_lower": bootstrap_results['ci_lower'],
		"ci_upper": bootstrap_results['ci_upper'],
		"missing": missing_count,
		"malformed_transcriptions": len(malformed_transcriptions)
	}


def main():
	args = parse_args()
	# normalize random_crop_secs
	args.random_crop_secs = _parse_random_crop(args.random_crop_secs)

	# Enforce offline mode and cache paths
	os.environ['HF_HUB_OFFLINE'] = '1'
	cache_dir = '/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/cache'
	os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')
	os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_dir, 'huggingface')
	os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')

	# Read original labels CSV
	try:
		df = pd.read_csv(args.test_labels_path)
	except Exception as exc:
		print(f"Failed to read labels from {args.test_labels_path}: {exc}")
		raise

	print(f"Loaded {len(df)} rows from {args.test_labels_path}")

	# Load device and model once
	device = set_device()
	net, params, checkpoint = load_model(device, args.checkpoint)

	# Ensure params has expected attributes
	if not hasattr(params, "evaluation_batch_size") or params.evaluation_batch_size is None:
		params.evaluation_batch_size = args.evaluation_batch_size or 1

	results = []
	total_expected = 0
	total_missing = 0
	total_malformed_transcriptions = 0
	for split in ["Test1", "Test2", "Test3"]:
		res = evaluate_split(split, df, net, params, device, args, LABELS_TO_IDS, args.transcriptions_dir)
		if res is not None:
			results.append(res)
			total_expected += len(df[df["Split_Set"] == split])
			total_missing += res.get("missing", 0)
			total_malformed_transcriptions += res.get("malformed_transcriptions", 0)

	print("Summary:")
	for r in results:
		if r["f1_macro"] is None:
			f1_display = "N/A"
			ci_display = ""
		else:
			f1_display = f"{r['f1_macro']:.4f}"
			ci_display = f" (95% CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}])"
		print(f"  {r['split']}: n={r['n']}  F1_macro={f1_display}{ci_display}")

	if total_expected > 0:
		overall_missing_pct = 100.0 * total_missing / total_expected
		print(f"Overall missing files: {total_missing}/{total_expected} ({overall_missing_pct:.2f}%)")
		print(f"Overall discarded malformed transcriptions: {total_malformed_transcriptions}")


if __name__ == "__main__":
	main()

"""_summary_

This script calculates the F-Score for the test set.

The labels_consensus.csv head is of the following format:
FileName,EmoClass,EmoAct,EmoVal,EmoDom,SpkrID,Gender,Split_Set
MSP-PODCAST_0001_0008.wav,N,2.2,4.0,2.6,30,Male,Test1
MSP-PODCAST_0001_0009.wav,N,3.777778,3.555556,3.777778,39,Male,Test1
MSP-PODCAST_0001_0011.wav,N,4.125,4.75,4.0,39,Male,Test1
MSP-PODCAST_0001_0013.wav,N,3.0,3.75,3.166667,39,Male,Test1
MSP-PODCAST_0001_0016.wav,N,2.4,4.2,2.8,30,Male,Test1

Where we can find different Split_Set but we will just focus on Test1, Test2, and Test3.
"""
