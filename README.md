# Real-Time Surgical Video Analysis

**High-accuracy semantic segmentation and surgical instrument tracking for laparoscopic surgery.**

This repository accompanies the thesis *"Real-Time Surgical Video Analysis: High-Accuracy Segmentation and Instrument Tracking"*. It contains two contributions for laparoscopic cholecystectomy video analysis:

1. **Efficient U-Net variant** — depthwise separable convolutions (DSC) combined with an instrument-specific lightweight attention mechanism and a composite loss function, achieving high accuracy at real-time speed.
2. **MambaMOT** — to our knowledge the first application of the Mamba State Space Model to surgical instrument tracking, maintaining instrument identity through occlusion.


---

## Highlights

| Task | Dataset | Key metric | Result |
|------|---------|-----------|--------|
| Segmentation | CholecSeg8k | Overall accuracy | **98.32 %** |
| Segmentation | CholecSeg8k | Instrument mean Dice | **0.921** |
| Segmentation | CholecSeg8k | Speed | **176.67 FPS** |
| Tracking | CholecTrack20 | HOTA | **84.87 %** |
| Tracking | CholecTrack20 | AssA | **87.43 %** |

- DSC reduces segmentation compute by **~7.8×** versus standard convolutions.
- The instrument-specific attention module cuts attention compute by **60 %** while improving instrument detection by **2.4 points**.
- MambaMOT improves Association Accuracy from **64.1 % → 87.43 %** over the previous best (SurgiTrack) and reduces the ID-switch rate by **~65 %**.

---

## Method overview

### 1. Segmentation model

A U-Net backbone (5-stage encoder–decoder with skip connections) extended with three components:

- **Depthwise separable convolutions (DSC)** applied systematically across all convolution blocks for efficiency.
- **Instrument-specific lightweight attention** — specialized filters tuned to the visual characteristics of surgical instruments (metallic reflection, edges, elongated shape).
- **Composite loss** — cross-entropy + Dice + an instrument-specific focal term (an extension of Focal Loss) to handle extreme class imbalance, with asymmetric class weighting (5.0× for instrument classes, 0.1× for background).

Output: per-pixel probabilities over **13 classes** (background, abdominal wall, liver, GI tract, fat, grasper, connective tissue, blood, cystic duct, L-hook electrocautery, gallbladder, hepatic vein, liver ligament).

### 2. MambaMOT (tracking)

A tracking-by-detection pipeline with four components:

1. **Detection** — YOLO11x fine-tuned on surgical video (7 instrument categories, 640×640 input).
2. **Feature extraction** — a custom module fusing visual, positional, and class information.
3. **Temporal modeling** — Mamba State Space Model; its input-dependent selective mechanism retains instrument features as state during occlusion and re-matches on reappearance.
4. **Association** — class-constrained two-stage association, built on ByteTrack with surgical-tracking-specific modifications.

---

## Results

### Tracking comparison (CholecTrack20, Intraoperative perspective)

| Method | HOTA | DetA | AssA | MOTA | IDF1 |
|--------|------|------|------|------|------|
| SORT | 31.2 | 35.4 | 27.5 | 28.1 | 33.8 |
| DeepSORT | 35.8 | 38.2 | 33.6 | 32.5 | 38.2 |
| ByteTrack | 40.3 | 42.5 | 38.2 | 38.7 | 43.5 |
| BoT-SORT | 41.8 | 43.1 | 40.5 | 40.2 | 45.1 |
| SMILETrack | 42.1 | 44.2 | 40.1 | 41.3 | 45.8 |
| SurgiTrack-SSL | 62.8 | 71.7 | 55.3 | – | 78.0 |
| SurgiTrack-FSL | 67.3 | 70.8 | 64.1 | – | – |
| **MambaMOT (ours)** | **84.87** | **82.39** | **87.43** | **70.06** | **90.34** |

See the thesis for full segmentation results, ablation studies, and per-sequence breakdowns.

---

## Installation

Requires Python 3.10+ and a CUDA-capable GPU (experiments used an NVIDIA A100). `mamba-ssm` and `causal-conv1d` require a matching CUDA toolkit — install the PyTorch build for your CUDA version first.

```bash
# 1. Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch torchvision

# 4. Install remaining dependencies
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` sometimes need `--no-build-isolation`:

```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

For tracking evaluation (HOTA, etc.), install [TrackEval](https://github.com/JonathonLuiten/TrackEval) separately.

---

## Datasets

Datasets are **not** included in this repository. Download them from the official sources and place them under `data/`.

- **CholecSeg8k** (segmentation) — 8,080 annotated frames from 17 cholecystectomy cases, 13 classes. Available on Kaggle / from the CAMMA research group.
- **CholecTrack20** (tracking) — 20 cholecystectomy cases, ~35,000 frames, 65,000 instrument instances, 7 categories. Request access from the dataset authors.

Expected layout:

```
data/
├── CholecSeg8k/
│   ├── train/
│   ├── val/
│   └── test/
└── CholecTrack20/
    ├── videos/
    └── annotations/
```

Please follow each dataset's license and usage terms.

---

## Usage

> Adjust the commands and paths below to match your actual scripts.

```bash
# Train the segmentation model
python segmentation/train.py --config configs/seg.yaml

# Evaluate segmentation
python segmentation/evaluate.py --weights checkpoints/seg_best.pth

# Train MambaMOT (detector first, then the Mamba + Re-ID modules)
python tracking/train_detector.py --config configs/yolo.yaml
python tracking/train_mambamot.py --config configs/track.yaml

# Run tracking inference on a video
python tracking/inference.py --video path/to/video.mp4 --weights checkpoints/mambamot.pth
```

### Key hyperparameters

**Segmentation:** input 384×384, AdamW, lr = 2e-4, β = (0.9, 0.999), weight decay = 1e-4, 50 epochs, mixed-precision training. Augmentations: horizontal/vertical flip, 90° rotation, brightness/contrast, Gaussian blur, color jitter (p = 0.1–0.5).

**Tracking:** YOLO11x detector; Mamba + Re-ID head trained with batch size 8 for 20 epochs. Mamba: 3 blocks, d_model = 256, d_state = 16, d_conv = 4, expand = 2. Association: τ_age = 3, τ_high = 0.5, τ_low = 0.1, τ_max = 30, Re-ID weight λ = 0.35, IoU threshold = 0.1.

---

## Project structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── .gitattributes
├── configs/            # YAML configs for training/eval
├── data/               # datasets (gitignored — download separately)
├── segmentation/       # segmentation model, training, evaluation
├── tracking/           # MambaMOT: detector, feature extraction, Mamba, association
├── checkpoints/        # model weights (gitignored)
└── results/            # output figures, metrics, visualizations
```

---

## Citation

If you use this code, please cite:

```bibtex
@thesis{miyoshi2025surgical,
  title  = {Real-Time Surgical Video Analysis: High-Accuracy Segmentation and Instrument Tracking},
  author = {Miyoshi, masanari},
  year   = {2025},
  school = {<Doshisha University>},
  type   = {Bachelor's thesis}
}


---

## License

Released under the MIT License — see [LICENSE](LICENSE). 
---

## Acknowledgments

This work builds on the CholecSeg8k and CholecTrack20 datasets (CAMMA research group) and on prior work including U-Net, Mamba, YOLO11, and ByteTrack. Thanks to the thesis advisor and examining committee for their feedback.
