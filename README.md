# A Cross-guided and Dynamic Adaptive Convolution Network for Ancient City Wall Defect Detection**

---

## Abstract

Defect detection on ancient architectural surfaces is critical for structural prevention and maintenance. However, existing detection methods perform poorly when handling complex textured backgrounds and diverse defect characteristics. To address these challenges, this paper proposes a defect detection network for ancient city walls based on cross-guidance and dynamic adaptive convolution.

- **CADSM (Cross-guided Attention Dual-Stream Module):** Tackles weak defect recognition under complex textures via an initial-prediction-guided foreground-background separation strategy with bidirectional guidance, effectively suppressing background noise interference.
- **DGBC (Dynamic Gated Bottleneck Convolution):** Extends the prior DynamicBottConv unit into a macro convolutional routing architecture. A data-driven gating system adaptively fuses deep and multi-scale features to handle defects of varying shapes.
- **WIoU Loss:** Introduced as a key optimization strategy. Its dynamic non-monotonic focusing mechanism reduces harmful gradients from low-quality anchors, mitigating regression oscillation caused by boundary ambiguity in ancient architectural datasets.

Experiments on the self-built **ZSHCityWall** dataset and the public **CMHB** dataset show that, compared to the baseline YOLOv11s, our method achieves **+3.4% Precision** and **+2.5% mAP50** on ZSHCityWall, and **+4.0% Precision** and **+1.7% mAP50** on CMHB.

---

## Installation

**Requirements**

- Python 3.9
- PyTorch 2.5.1 (cu124)
- CUDA 12.6
- cuDNN 8.9.4

```bash
# Create conda environment
conda create -n zshv11 python=3.9
conda activate zshv11

# Install PyTorch (cu124)
pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ultralytics dependencies
pip install ultralytics
```

---

## Dataset

### ZSHCityWall (Self-built)

A dataset of ancient city wall surface defects covering four categories:

| ID | Class | Description |
|----|-------|-------------|
| 0 | Cavity | Surface cavities and holes |
| 1 | Crack | Surface cracks |
| 2 | Efflorescence | Salt weathering / efflorescence |
| 3 | Erosion | Surface erosion |

Dataset available at: [https://www.kaggle.com/datasets/zsh666zsh/zshcitywall/data](https://www.kaggle.com/datasets/zsh666zsh/zshcitywall/data)

### CMHB (Public)

Public dataset for cultural heritage and masonry building defect detection.

### Directory Structure

Organize your data in YOLO format:

```
datasets/
├── ZSHCityWall/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── CMHB/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

## Training

```bash
python train.py \
    --model ultralytics/cfg/models/ZSH_Yaml/yolo11_CADSAM_DGBC.yaml \
    --data datasets/ZSHCityWall.yaml \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --lr0 0.01 \
    --weight-decay 0.0005 \
    --optimizer SGD \
    --workers 12
```

**Training Configuration Summary**

| Parameter | Value |
|-----------|-------|
| Input size | 640 × 640 |
| Epochs | 300 |
| Batch size | 16 |
| Initial LR | 0.01 |
| Weight decay | 0.0005 |
| Optimizer | SGD |
| Workers | 12 |

**Hardware:** NVIDIA GeForce RTX 4090 · Intel Core i9-14900KF · Windows 10
**Framework:** PyTorch 2.5.1 cu124 · Anaconda3 virtual environment

---

## Evaluation

```bash
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data datasets/ZSHCityWall.yaml \
    --imgsz 640
```

**Evaluation Metrics**

- Precision (P)
- Recall (R)
- mAP@50
- mAP@50-95

---

## Core Source Code

Key implementation files:

- **CADSM:** `ultralytics/nn/ZSH_Add/CADSAM.py`
- **DGBC:** `ultralytics/nn/ZSH_Add/DGBC.py`
- **WIoU:** `ultralytics/utils/loss.py`
- **Model YAML:** `ultralytics/cfg/models/ZSH_Yaml/yolo11_CADSAM_DGBC.yaml`

GitHub repository: [https://github.com/ZSH666zsh/zshv11-github-main](https://github.com/ZSH666zsh/zshv11-github-main)

---

## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{zsh2025citywall,
  title={Ancient City Wall Defect Detection Based on Cross-guided and Dynamic Adaptive Convolution},
  author={Shihang Zhao, Zengxin Chen, Yage Zhang, Yijie Guan, Yongbo Yu, Jianwei Yue},
  journal={},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

This repository is built upon [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics). We gratefully acknowledge their open-source contribution.
