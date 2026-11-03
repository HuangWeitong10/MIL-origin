import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import EffusionMIL
from pipeline.data import aggregate, load_bag, read_manifest, validate_patient_splits


def evaluate(model, rows, device, feature_dim, num_classes):
    model.eval()
    probabilities = []
    losses = []
    with torch.no_grad():
        for row in rows:
            bag = load_bag(row["bag_path"], feature_dim).to(device)
            output = model(bag)
            probabilities.append(output["Y_prob"].squeeze(0).cpu().numpy())
            label = torch.tensor([row["label"]], device=device)
            losses.append(F.cross_entropy(output["logits"], label).item())
    slide_scores, patient_scores = aggregate(rows, probabilities)
    labels = {row["patient_id"]: row["label"] for row in rows}
    slide_labels = {(row["patient_id"], row["slide_id"]): row["label"] for row in rows}
    slide_acc = np.mean([score.argmax() == slide_labels[key] for key, score in slide_scores.items()])
    ids = sorted(patient_scores)
    truth = np.asarray([labels[patient] for patient in ids])
    score = np.stack([patient_scores[patient] for patient in ids])
    aucs = []
    for class_id in range(num_classes):
        binary = (truth == class_id).astype(int)
        if binary.min() != binary.max():
            aucs.append(roc_auc_score(binary, score[:, class_id]))
    return {
        "bag_loss": float(np.mean(losses)),
        "patient_acc": float(np.mean(score.argmax(axis=1) == truth)),
        "slide_acc": float(slide_acc),
        "patient_macro_auc": float(np.mean(aucs)) if aucs else None,
        "patients": len(ids),
        "bags": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the MIL model from trusted feature bags")
    parser.add_argument("--manifest", required=True, help="CSV: bag_path,patient_id,slide_id,label,split")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--k-sample", type=int, default=16)
    parser.add_argument("--bag-weight", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.bag_weight <= 1:
        parser.error("--bag-weight must be in [0, 1]")

    rows = read_manifest(args.manifest, require_labels=True)
    validate_patient_splits(rows)
    train_rows = [row for row in rows if row["split"].lower() == "train"]
    val_rows = [row for row in rows if row["split"].lower() == "val"]
    if not train_rows or not val_rows:
        parser.error("The manifest needs both train and val rows")
    if any(not 0 <= row["label"] < args.classes for row in rows):
        parser.error("Labels must be in [0, classes)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    feature_dim = load_bag(train_rows[0]["bag_path"]).shape[1]
    config = {"embed_dim": feature_dim, "n_classes": args.classes,
              "k_sample": args.k_sample, "subtyping": True,
              "instance_eval": True, "size_arg": "big", "dropout": 0.15}
    model = EffusionMIL(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = float("-inf")
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        train_losses = []
        for index in order:
            row = train_rows[index]
            bag = load_bag(row["bag_path"], feature_dim).to(device)
            label = torch.tensor([row["label"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            result = model(bag, label=label, instance_eval=True)
            bag_loss = F.cross_entropy(result["logits"], label)
            loss = args.bag_weight * bag_loss + (1 - args.bag_weight) * result["instance_loss"]
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        metrics = evaluate(model, val_rows, device, feature_dim, args.classes)
        metrics.update(epoch=epoch, train_loss=float(np.mean(train_losses)))
        history.append(metrics)
        auc = metrics["patient_macro_auc"]
        candidate = auc if auc is not None else -metrics["bag_loss"]
        if candidate > best_auc or (candidate == best_auc and metrics["bag_loss"] < best_loss):
            best_auc, best_loss = candidate, metrics["bag_loss"]
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "metrics": metrics}, output / "best.pt")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
