"""Cut an OpenSlide or SDPC whole slide into ordered image bags."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def open_slide(path, backend):
    if backend == "openslide":
        import openslide

        return openslide.OpenSlide(str(path))
    if backend == "sdpc":
        import sdpc

        return sdpc.Sdpc(str(path))
    raise ValueError(f"Unknown backend: {backend}")


def cut_slide(slide, slide_id, output, level, patch_size, stride, bag_size, white_fraction):
    width, height = slide.level_dimensions[level]
    width0, height0 = slide.level_dimensions[0]
    scale_x, scale_y = width0 / width, height0 / height
    slide_dir = Path(output) / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            x0, y0 = round(x * scale_x), round(y * scale_y)
            patch = Image.fromarray(np.asarray(slide.read_region((x0, y0), level,
                                                                  (patch_size, patch_size)))).convert("RGB")
            pixels = np.asarray(patch)
            blank = np.mean(np.all(pixels >= 235, axis=2))
            if blank > white_fraction:
                continue
            index = len(rows)
            bag_id = f"bag_{index // bag_size:04d}"
            patch_dir = slide_dir / bag_id
            patch_dir.mkdir(exist_ok=True)
            patch_path = patch_dir / f"{index:06d}_x{x0}_y{y0}.jpg"
            patch.save(patch_path, quality=95)
            rows.append([bag_id, index % bag_size, str(patch_path.relative_to(slide_dir)), x0, y0, level])
    with (slide_dir / "patches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bag_id", "patch_index", "patch_path", "x_level0", "y_level0", "level"])
        writer.writerows(rows)
    return len(rows), (len(rows) + bag_size - 1) // bag_size


def main():
    parser = argparse.ArgumentParser(description="Cut a whole slide into image bags")
    parser.add_argument("--wsi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["openslide", "sdpc"], default="openslide")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--bag-size", type=int, default=64)
    parser.add_argument("--white-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if min(args.patch_size, args.stride, args.bag_size) < 1 or not 0 <= args.white_fraction <= 1:
        parser.error("Sizes must be positive and --white-fraction must be in [0, 1]")
    slide = open_slide(args.wsi, args.backend)
    try:
        patches, bags = cut_slide(slide, Path(args.wsi).stem, args.output, args.level,
                                  args.patch_size, args.stride, args.bag_size, args.white_fraction)
    finally:
        close = getattr(slide, "close", None)
        if close is not None:
            close()
    print(f"Saved {patches} patches in {bags} bags")


if __name__ == "__main__":
    main()
