"""Extract one UNI feature tensor per image bag."""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Extract UNI patch features")
    parser.add_argument("--patch-root", required=True, help="Root produced by cut_patches.py")
    parser.add_argument("--weights", required=True, help="Locally obtained UNI checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    import timm
    from torchvision import transforms

    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                              init_values=1e-5, num_classes=0)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    model.load_state_dict(weights, strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    root = Path(args.patch_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for slide_dir in sorted(root.iterdir()):
        if not slide_dir.is_dir():
            continue
        for bag_dir in sorted(slide_dir.glob("bag_*")):
            patches = sorted(bag_dir.glob("*.jpg"))
            if not patches:
                continue
            features = []
            with torch.inference_mode():
                for start in range(0, len(patches), args.batch_size):
                    batch = []
                    for patch in patches[start:start + args.batch_size]:
                        with Image.open(patch) as image:
                            batch.append(transform(image.convert("RGB")))
                    features.append(model(torch.stack(batch).to(device)).cpu())
            tensor = torch.cat(features).float()
            feature_dir = output / slide_dir.name
            feature_dir.mkdir(exist_ok=True)
            feature_path = feature_dir / f"{bag_dir.name}.pt"
            torch.save(tensor, feature_path)
            index_rows.append([str(feature_path), slide_dir.name, bag_dir.name, len(patches)])
    with (output / "feature_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bag_path", "slide_id", "bag_id", "num_patches"])
        writer.writerows(index_rows)
    print(f"Saved {len(index_rows)} feature bags")


if __name__ == "__main__":
    main()
