# Visiope Object Detection ⚡🎯

**Optimized YOLO variants for faster inference with maintained accuracy**

---

## Overview

This project implements **lightweight versions** of classic YOLO architectures to reduce inference time while preserving detection accuracy. We developed two compact variants and compared them against YOLOv8.

**Goal:** Optimize inference speed without significant accuracy loss

## Implemented Models

- **YOLO v1 Tiny** - Compact version of original YOLO architecture
- **YOLO v2 Tiny** - Lightweight YOLOv2 variant
- **YOLOv8** - Baseline comparison model

## Results

Our optimized variants achieved:
- **Reduced inference time** compared to standard models
- **Maintained detection accuracy** on test datasets
- **Efficient architecture** suitable for real-time applications

## Key Files

- `models/yolo.py` - YOLO v1 and v2 Tiny implementations
- `YOLO_v8.ipynb` - YOLOv8 baseline experiments
- `yolo_v2_tiny.ipynb` - YOLO v2 Tiny development
- `train.py` - Training pipeline
- `test.py` - Evaluation and benchmarking

## Usage

```bash
# Train models
python train.py

# Run inference comparison
python test.py

# View experiments
jupyter notebook main.ipynb
```

---

**Technologies:** PyTorch, YOLO, Object Detection, Model Optimization