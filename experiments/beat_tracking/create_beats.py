import os
import sys
import mir_eval
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple

# Add project root to the path
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dirs import METADATA_DIR, RAW_DATA_DIR, EXPERIMENTS_DIR

# Configuration constants
ERROR_RATIO = 0.3
DELETE_RATIO = 0.3
INSERT_RATIO = 0.4
OFFSET_RATIO = 0.3
NOISY_RATIO = 0.4
MAX_BPM = 1000
MAX_INTERVAL = 60 / MAX_BPM


def create_beats_pred(beats_anno: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Create a predicted beats array with artificial errors.

    Parameters:
        beats_anno (np.ndarray): Original beat annotations.

    Returns:
        Tuple[np.ndarray, float]: Modified beats with errors and the F1 score.
    """
    beats_pred = np.array(beats_anno, dtype=np.float64)

    beats_pred = delete_beats(beats_pred, DELETE_RATIO)
    beats_pred = insert_beats(beats_pred, INSERT_RATIO)
    beats_pred = apply_offset_errors(beats_pred, OFFSET_RATIO)
    beats_pred = add_noise(beats_pred, NOISY_RATIO)

    beats_pred = np.sort(beats_pred)
    return check_beats(beats_pred, beats_anno)


def delete_beats(beats_pred: np.ndarray, ratio: float) -> np.ndarray:
    delete_num = int(len(beats_pred) * ERROR_RATIO * ratio)
    delete_indices = np.random.choice(len(beats_pred), delete_num, replace=False)
    return np.delete(beats_pred, delete_indices)


def insert_beats(beats_pred: np.ndarray, ratio: float) -> np.ndarray:
    insert_num = int(len(beats_pred) * ERROR_RATIO * ratio)
    insert_indices = np.random.choice(len(beats_pred) - 1, insert_num, replace=False)
    insert_beats = (beats_pred[insert_indices] + beats_pred[insert_indices + 1]) / 2
    return np.insert(beats_pred, insert_indices + 1, insert_beats)


def apply_offset_errors(beats_pred: np.ndarray, ratio: float) -> np.ndarray:
    offset_num = int(len(beats_pred) * ERROR_RATIO * ratio)
    offset_indices = np.random.choice(len(beats_pred), offset_num, replace=False)
    offset_values = np.random.rand(offset_num) * 0.2 + 0.07
    offset_signs = np.random.choice([-1, 1], offset_num)
    beats_pred[offset_indices] += offset_values * offset_signs
    return beats_pred


def add_noise(beats_pred: np.ndarray, ratio: float) -> np.ndarray:
    noisy_num = int(len(beats_pred) * ratio)
    noisy_indices = np.random.choice(len(beats_pred), noisy_num, replace=False)
    noisy_values = np.random.rand(noisy_num) * 0.14 - 0.07
    beats_pred[noisy_indices] += noisy_values
    return beats_pred


def check_beats(
    beats_pred: np.ndarray, beats_anno: np.ndarray
) -> Tuple[np.ndarray, float]:
    delete_indices = [
        i
        for i in range(1, len(beats_pred))
        if beats_pred[i] - beats_pred[i - 1] < MAX_INTERVAL
    ]
    beats_pred = np.delete(beats_pred, delete_indices)
    beats_pred = beats_pred[beats_pred >= 0]

    beats_pred_trimmed = mir_eval.beat.trim_beats(beats_pred)
    beats_annos_trimmed = mir_eval.beat.trim_beats(beats_anno)
    f1_score = mir_eval.beat.f_measure(beats_annos_trimmed, beats_pred_trimmed)
    return beats_pred, f1_score


def main():
    split = "test"
    beats_pred_dir = os.path.join(
        EXPERIMENTS_DIR, "beat_tracking/beats_with_error", split
    )
    beats_pred_img_dir = os.path.join(beats_pred_dir, "img")
    os.makedirs(beats_pred_dir, exist_ok=True)
    os.makedirs(beats_pred_img_dir, exist_ok=True)

    metadata_path = os.path.join(METADATA_DIR, f"{split}_metadata.csv")
    metadata = pd.read_csv(metadata_path)
    perf_ids = metadata["performance_id"].tolist()
    f1_scores = []

    for perf_id in tqdm(perf_ids):
        perf_id_short = "_".join(perf_id.split("_")[:2])
        row = metadata[metadata["performance_id"] == perf_id_short]
        anno_path = os.path.join(
            RAW_DATA_DIR,
            "ACPAS-dataset",
            row["folder"].iloc[0],
            row["performance_annotation"].iloc[0],
        )

        try:
            with open(anno_path, "r") as f:
                beats_anno = np.array(
                    [float(line.split("\t")[0]) for line in f], dtype=np.float64
                )
        except Exception as e:
            print(f"Error reading annotation file {anno_path}: {e}")
            continue

        beats_pred, f1_score = create_beats_pred(beats_anno)
        f1_scores.append(f1_score)

        beats_pred_path = os.path.join(beats_pred_dir, f"{perf_id}.npy")
        np.save(beats_pred_path, beats_pred)

    print(f"Mean F1 score: {np.mean(f1_scores)}")


if __name__ == "__main__":
    main()
