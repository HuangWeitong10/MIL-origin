import argparse
import csv
from pathlib import Path

import torch

from pipeline.data import aggregate, load_bag, read_manifest
from pipeline.models.effusion_mil import EffusionMIL


def write_scores(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Predict feature bags and aggregate to slides/patients")
    parser.add_argument("--manifest", required=True, help="CSV: bag_path,patient_id,slide_id")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = EffusionMIL(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probabilities = []
    with torch.no_grad():
        for row in rows:
            bag = load_bag(row["bag_path"], config["embed_dim"]).to(device)
            probabilities.append(model(bag)["Y_prob"].squeeze(0).cpu().tolist())

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scores = [f"class_{i}" for i in range(config["n_classes"])]
    write_scores(output / "bags.csv", ["bag_path", "patient_id", "slide_id", *scores],
                 [[row["bag_path"], row["patient_id"], row["slide_id"], *prob]
                  for row, prob in zip(rows, probabilities, strict=True)])
    slides, patients = aggregate(rows, probabilities)
    write_scores(output / "slides.csv", ["patient_id", "slide_id", *scores],
                 [[patient, slide, *prob] for (patient, slide), prob in sorted(slides.items())])
    write_scores(output / "patients.csv", ["patient_id", *scores],
                 [[patient, *prob] for patient, prob in sorted(patients.items())])
    print(f"Saved {len(rows)} bags, {len(slides)} slides, {len(patients)} patients to {output}")


if __name__ == "__main__":
    main()
