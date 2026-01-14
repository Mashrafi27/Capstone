# Optimizing Vision-Based Object Detection and Lane Segmentation for ADAS

Multi-task vision stack that combines object detection and lane segmentation for advanced driver assistance systems (ADAS), with a focus on optimization for resource-constrained devices. The implementation builds on YOLO-style detection and CLRNet-style lane detection, exploring multi-task optimization strategies such as cross-stitching and gradient balancing.

Report: `report.pdf`  
Poster: `poster.pdf`

![Capstone poster](poster.png)

## Project Summary
- Joint training for detection + lane segmentation to share features and reduce compute.
- Experiments with cross-stitch layers and PCGrad/FAMO-style conflict handling.
- Evaluation artifacts and logs live under `runs/` and `scores/`.

## Repository Structure
- `main.py`: main experiment entry point used during development.
- `mains/`: alternative training scripts for different multi-task setups.
- `multitask.py`, `multitasks/`: multi-task model variants and utilities.
- `clrnet/`, `nets/`, `utils/`, `losses/`: core model components and training helpers.
- `configs/`: CLRNet configuration files.
- `runs/`, `scores/`: experiment outputs and summaries.

## Environment
Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
The training pipeline expects datasets like BDD100K (object detection) and CULane (lane segmentation). Paths are configured in the scripts and config files under `configs/` and `utils/`.

## Training and Evaluation
Start from `main.py`, or use the variants under `mains/` to reproduce specific experiments:
```bash
python main.py
```
Adjust configs and checkpoints in `configs/` and the relevant `mains/*.py` script before running.

## Artifacts
The poster and report are included for quick project context. The presentation PDF is intentionally ignored via `.gitignore`.
