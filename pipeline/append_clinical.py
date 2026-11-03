"""Append age, sex and sampling site to each UNI patch vector."""

import argparse
import csv
from pathlib import Path

import torch

from pipeline.data import load_bag, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Create [N, 1030] feature bags")
    parser.add_argument("--manifest", required=True,
                        help="CSV with bag_path,patient_id,slide_id,gender,age,site_index")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    required = {"gender", "age", "site_index"}
    if required - rows[0].keys():
        parser.error(f"Missing clinical columns: {sorted(required - rows[0].keys())}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    produced = []
    for index, row in enumerate(rows):
        gender = float(row["gender"])
        age = float(row["age"])
        site = int(row["site_index"])
        if gender not in {0, 1} or not 0 <= age <= 120 or site not in {0, 1, 2}:
            raise ValueError(f"Invalid clinical fields in row {index + 1}")
        bag = load_bag(row["bag_path"], 1024)
        clinical = torch.tensor([1 - gender, gender, age / 100,
                                 *(int(site == i) for i in range(3))], dtype=bag.dtype)
        combined = torch.cat([bag, clinical.expand(len(bag), -1)], dim=1)
        path = output / f"bag_{index:06d}.pt"
        torch.save(combined, path)
        produced.append({**row, "bag_path": str(path)})
    columns = list(produced[0])
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(produced)
    print(f"Saved {len(produced)} feature bags with clinical data")


if __name__ == "__main__":
    main()
