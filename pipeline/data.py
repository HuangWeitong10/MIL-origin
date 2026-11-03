import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def read_manifest(path, require_labels=False):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"bag_path", "patient_id", "slide_id"}
        if require_labels:
            required |= {"label", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    for row in rows:
        bag_path = Path(row["bag_path"])
        row["bag_path"] = str(bag_path if bag_path.is_absolute() else path.parent / bag_path)
        if not row["patient_id"] or not row["slide_id"]:
            raise ValueError("Every bag needs patient_id and slide_id")
        if require_labels:
            row["label"] = int(row["label"])
    return rows


def validate_patient_splits(rows):
    patient_split = {}
    patient_label = {}
    for row in rows:
        patient = row["patient_id"]
        split = row["split"].lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split: {split}")
        if patient in patient_split and patient_split[patient] != split:
            raise ValueError(f"Patient {patient} appears in multiple splits")
        if patient in patient_label and patient_label[patient] != row["label"]:
            raise ValueError(f"Patient {patient} has conflicting labels")
        patient_split[patient] = split
        patient_label[patient] = row["label"]


def load_bag(path, expected_dim=None):
    # Only load feature files produced by a trusted source.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1:
        raise ValueError(f"Expected nonempty [patch, feature] tensor: {path}")
    if expected_dim is not None and value.shape[1] != expected_dim:
        raise ValueError(f"Feature dimension mismatch in {path}: {value.shape[1]} != {expected_dim}")
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite feature values in {path}")
    return value


def aggregate(rows, probabilities):
    """Average bags within slides, then slides equally within each patient."""
    slides = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        slides[(row["patient_id"], row["slide_id"])].append(np.asarray(probability))
    slide_scores = {key: np.mean(values, axis=0) for key, values in slides.items()}
    patients = defaultdict(list)
    for (patient, _), probability in slide_scores.items():
        patients[patient].append(probability)
    patient_scores = {key: np.mean(values, axis=0) for key, values in patients.items()}
    return slide_scores, patient_scores
