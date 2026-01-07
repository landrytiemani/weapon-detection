# Modular Real-Time Weapon Detection Framework

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**A Lightweight Modular Real-Time Weapon Detection Framework Using Advanced Computer Vision Techniques for Edge Deployment Optimization**

[Overview](#abstract) •
[Architecture](#architecture) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Results](#results) •
[Citation](#citation)

</div>

---

## Abstract

This repository contains the complete implementation for doctoral dissertation research at **Harrisburg University of Science and Technology**. The framework achieves significant improvements in weapon detection through a novel **two-stage hierarchical detection pipeline** with **ByteTrack temporal tracking** integration.

### Key Achievements

| Metric | Improvement | Description |
|--------|-------------|-------------|
|  **mAP@0.5** | +46% | Over single-stage baselines |
|  **False Positives** | -71% | Through temporal tracking |
| **Real-time** | >30 FPS | On edge devices |
| **Privacy** | <10% overhead | Selective face blurring |

---

## Key Contributions

| # | Contribution | Research Question |
|---|--------------|-------------------|
| 1 | **Modular Pipeline Architecture** - Two-stage hierarchical detection with configurable components | RQ1 |
| 2 | **Architecture Comparison** - RT-DETR (Transformer) vs YOLOv8-EfficientViT (CNN) | RQ2 |
| 3 | **Temporal Tracking Integration** - ByteTrack for FP reduction and ID consistency | RQ3 |
| 4 | **Privacy-Preserving Mechanisms** - Selective face blurring with minimal accuracy impact | RQ4 |

---

## Architecture

```

┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODULAR WEAPON DETECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐     ┌─────────────────────┐     ┌────────────────────────┐   │
│  │  INPUT   │     │      STAGE 1        │     │       STAGE 2          │   │
│  │  FRAME   │────▶│  Person Detection   │────▶│   Weapon Detection     │   │
│  │          │     │   + ByteTrack       │     │   (RT-DETR/EffViT)     │   │
│  └──────────┘     └─────────────────────┘     └────────────────────────┘   │
│                            │                            │                   │
│                            ▼                            ▼                   │
│                   ┌─────────────────┐         ┌──────────────────┐         │
│                   │ Person Crops    │         │ Post-Processing  │         │
│                   │ • Scale: 2.5×   │         │ • NMS (local)    │         │
│                   │ • Square        │         │ • NMS (global)   │         │
│                   │ • Overlap filter│         │ • Cross-class    │         │
│                   └─────────────────┘         └──────────────────┘         │
│                                                         │                   │
│                                                         ▼                   │
│                                                ┌──────────────────┐         │
│                                                │  Privacy Module  │         │
│                                                │  (Face Blurring) │         │
│                                                └──────────────────┘         │
│                                                         │                   │
│                                                         ▼                   │
│                                                ┌──────────────────┐         │
│                                                │     OUTPUT       │         │
│                                                │ Weapon Detections│         │
│                                                │ + Track IDs      │         │
│                                                └──────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Person Detection & Tracking

| Component | Model | GFLOPs | Purpose |
|-----------|-------|--------|---------|
| Detector | YOLOv8n | 8.7 | Real-time person detection |
| Alternative | SSD-MobileNetV2 | 3.4 | Lightweight option |
| Tracker | ByteTrack | ~0.1 | Multi-object tracking |

### Stage 2: Weapon Detection

| Architecture | Type | GFLOPs | mAP@0.5 | Best For |
|-------------|------|--------|---------|----------|
| **RT-DETR** | Transformer | 81.4 | **0.721** | Accuracy-critical |
| **YOLOv8-EfficientViT** | Transformer-CNN | 6.2 | 0.669 | Edge deployment |

---

## Research Questions

### RQ1: Modular Ablation Study
> How do individual pipeline components contribute to detection accuracy?

-  Crop scale 2.0-2.5× optimal for weapon visibility
- Overlap filtering reduces duplicates by >50%
- Test-time augmentation improves mAP by 3-5%

### RQ2: Architecture Comparison
> How does RT-DETR compare to YOLOv8-EfficientViT?

- RT-DETR achieves 5.2% higher mAP@0.5
- EfficientViT maintains 93% accuracy at 13× less compute
- Larger performance gap on knives vs handguns

### RQ3: Temporal Tracking
> How does ByteTrack affect detection quality?

- Tracking reduces false positives by 71%
- Frame gap 3-5 maintains ≥95% accuracy
- 17% speed improvement with gap=3

### RQ4: Privacy Preservation
> Can privacy protection be achieved with minimal impact?

- Selective face blurring adds only 8% latency
- Privacy maintains mAP within 2%
- Pixelation 5% faster than Gaussian blur

---

## Installation

### Prerequisites

- **OS**: Ubuntu 20.04+ / Windows 10+
- **Python**: 3.10+
- **GPU**: NVIDIA with 8GB+ VRAM
- **CUDA**: 11.8+

### Quick Install

```bash
# Clone repository
git clone https://github.com/landrytiemani/weapon-detection-pipeline.git
cd weapon-detection-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Download weights
bash scripts/download_weights.sh
```

### Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

from ultralytics import YOLO
print("✓ Installation successful!")
```

---

## Quick Start

### Run Inference

```bash
# Single experiment with tracking
python main_perclass.py --config config.yaml
```

### Run Research Experiments

```bash
# RQ1: Ablation Study
python RQ/run_rq1_ablation.py --config config.yaml

# RQ2: Architecture Comparison
python RQ/run_rq2_architecture.py --config config.yaml

# RQ3: Tracking Experiments
python RQ/run_rq3_tracking.py --config config.yaml

# RQ4: Privacy Analysis
python RQ/run_rq4_privacy.py --config config.yaml

# Compute GFLOPs
python RQ/compute_flops.py --results-dir Results/
```

---

## Project Structure

```
weapon-detection-pipeline/
│
├── main_perclass.py             #  Main entry point
├── config.yaml                  #  Primary configuration
├── requirements.txt             #  Dependencies
│
├── stages/                      # Pipeline stages
│   ├── stage_2_persondetection.py    # Person detection + tracking
│   └── stage_3_weapondetection.py    # Weapon detection
│
├── models/                      # Detection models
│   ├── person_detectors/
│   │   ├── yolov8_tracker.py         # YOLOv8 + ByteTrack
│   │   └── ssd_mobilenet_bytetrack.py
│   └── weapon_detectors/
│       ├── rt_detr.py                # RT-DETR (Transformer)
│       └── yolov8_efficientvit.py    # EfficientViT (CNN)
│
├── tracker/                     # ByteTrack implementation
│   ├── byte_tracker.py          # Main tracker
│   ├── kalman_filter.py         # Motion prediction
│   ├── matching.py              # Hungarian matching
│   └── basetrack.py             # Track base class
│
├── utils/                       # Utilities
│   ├── box_utils.py             # Bounding box operations
│   ├── evaluation.py            # mAP calculation
│   ├── privacy.py               # Face blurring module
│   ├── visualization.py         # Debug visualizations
│   ├── flops_utils.py           # GFLOPs computation
│   └── report_utils.py          # Report generation
│
├── vision/                      # SSD MobileNet support
│   ├── ssd/                     # SSD architecture
│   └── nn/                      # Neural network modules
│
├── RQ/                          # Research experiments
│   ├── run_rq1_ablation.py
│   ├── run_rq2_architecture.py
│   ├── run_rq3_tracking.py
│   ├── run_rq4_privacy.py
│   └── compute_flops.py
│
├── configs/                     # Configuration files
│   └── person_tracker_bytetrack.yaml
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   └── WEIGHTS.md
│
├── results/                     # Experiment results
│   └── figures/
│
└── notebooks/                   # Training notebooks
    └── Train.ipynb
```

---

## Configuration

### Main Configuration (`config.yaml`)

```yaml
pipeline:
  frames_dir: data/test/images
  labels_dir: data/test/labels

stage_2:
  approach: yolov8_tracker # or: sd_mobilenet_bytetrack
  crop_scale: 1.8
  use_tracker: true
  frame_gap: 1
  yolov8_tracker:
    model_path: weights/person/yolov8n.pt
    confidence_threshold: 0.15

stage_3:
  approach: yolov8_efficientvit  # or: rt_detr
  imgsz: 512
  nms_iou_threshold: 0.45
  names: ["handgun", "knife"]

privacy:
  enabled: true
  scope: "non_targets"
  face_blur:
    method: "pixelate"
```

---

## Results

### Overall Performance

| Configuration | mAP@0.5 | Precision | Recall | F1 | FPS |
|--------------|---------|-----------|--------|-----|-----|
| **RT-DETR + Tracking** | **0.721** | 0.847 | 0.681 | 0.755 | 24.3 |
| EfficientViT + Tracking | 0.669 | 0.812 | 0.634 | 0.712 | 38.7 |
| Baseline (single-stage) | 0.494 | 0.623 | 0.521 | 0.567 | 42.1 |

### Per-Class Performance

| Class | RT-DETR | EfficientViT | Δ |
|-------|---------|--------------|---|
| **Handgun** | 0.784 | 0.752 | +3.2% |
| **Knife** | 0.658 | 0.586 | +7.2% |

### Key Findings

1. **Two-stage approach** → +46% mAP over single-stage
2. **ByteTrack integration** → -71% false positives
3. **RT-DETR vs EfficientViT** → +5.2% mAP but 13× more compute
4. **Privacy protection** → Only 8% latency overhead

---

## Citation

If you use this code, please cite:

```bibtex
@phdthesis{tiemani2026weapon,
  title     = {A Lightweight Modular Real-Time Weapon Detection Framework Using 
               Advanced Computer Vision Techniques for Edge Deployment Optimization},
  author    = {Tiemani, Landry},
  year      = {2026},
  school    = {Harrisburg University of Science and Technology},
  type      = {Ph.D. Dissertation}
}
```

### Related Works

```bibtex
@inproceedings{zhang2022bytetrack,
  title     = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author    = {Zhang, Yifu and others},
  booktitle = {ECCV},
  year      = {2022}
}

@inproceedings{lv2024rtdetr,
  title     = {DETRs Beat YOLOs on Real-time Object Detection},
  author    = {Lv, Wenyu and others},
  booktitle = {CVPR},
  year      = {2024}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 & RT-DETR
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [EfficientViT](https://github.com/microsoft/Cream) - Efficient backbone
- Harrisburg University of Science and Technology

---

<div align="center">

**Developed for Ph.D. Dissertation in Data Sciences**

Harrisburg University of Science and Technology

*Landry Tiemani • Expected 2026*

</div>
