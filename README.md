# MIL-origin

Code for a six-class effusion cytopathology MIL workflow. The repository includes slide patching, UNI feature extraction, clinical feature assembly, MIL training, and bag/slide/patient inference. The model is a preliminary research implementation; additional code is coming soon.

## Workflow

| Step | Entry point | Output |
| --- | --- | --- |
| Cut whole-slide images | `pipeline/cut_patches.py` | JPEG bags and `patches.csv` with level-0 coordinates |
| Extract UNI features | `pipeline/extract_uni.py` | One `[N, 1024]` tensor per bag and `feature_index.csv` |
| Add clinical features | `pipeline/append_clinical.py` | One `[N, 1030]` tensor per bag |
| Train | `pipeline/train.py` | `best.pt`, `history.csv` |
| Infer | `pipeline/infer.py` | `bags.csv`, `slides.csv`, `patients.csv` |

Bag length `N` is variable. Patch order is preserved by zero-padded filenames, and `patches.csv` maps each feature row back to the source slide coordinate. The model accepts either 1024 or 1030 input features, depending on the training manifest.

## Setup

```bash
pip install -r requirements.txt
```

OpenSlide slides also need `openslide-python` and the OpenSlide system library. For SDPC slides, install the `sdpc` reader used by your environment and set `--backend sdpc`. UNI weights are supplied separately through `--weights`.

## Example commands

```bash
python -m pipeline.cut_patches --wsi slide.svs --output patches --backend openslide --level 0 --patch-size 224 --bag-size 64
python -m pipeline.extract_uni --patch-root patches --weights uni_weights.bin --output features --device cuda:0
python -m pipeline.append_clinical --manifest clinical_manifest.csv --output features_clinical
python -m pipeline.train --manifest features_clinical/manifest.csv --output runs/fold0 --classes 6 --device cuda:0
python -m pipeline.infer --manifest features_clinical/manifest.csv --checkpoint runs/fold0/best.pt --output predictions --device cuda:0
```

The training manifest has columns `bag_path,patient_id,slide_id,label,split`. Labels are integers `0` through `5`; split values are `train`, `val`, or `test`. One patient must appear in only one split. Paths can be absolute or relative to the manifest. The inference manifest only needs `bag_path,patient_id,slide_id`.

To make 1030-dimensional features, add `gender,age,site_index` columns to the input manifest. `gender` uses `0` for male and `1` for female; `age` is divided by 100; `site_index` uses `0` for pleural, `1` for ascitic, and `2` for pericardial fluid. These six values are appended to every patch vector. Include `label,split` in this manifest if it will be used for training.

The validation history reports slide and patient accuracy and patient one-vs-rest macro AUC. Inference averages bag probabilities within each slide, then averages slide probabilities within each patient.

## Source and data

The model and training workflow were adapted from the project's earlier MIL research code under the MIT license. The cutting and UNI extraction entries are streamlined versions of the earlier preprocessing workflow. No slide images, patient records, extracted features, pretrained weights, or training checkpoints are included.
