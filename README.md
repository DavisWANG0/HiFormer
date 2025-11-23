# Ultra-High-Definition Image Restoration via High-Frequency Enhanced Transformer (T-CSVT 2025)

[Chen Wu](https://github.com/5chen), [Ling Wang](https://daviswang0.github.io/), [Zhuoran Zheng](https://scholar.google.com.hk/citations?user=pXzPL-sAAAAJ&hl=zh-CN), [Weidong Jiang](https://xplorestaging.ieee.org/author/37288834600), [Yuning Cui](https://www.ce.cit.tum.de/en/air/people/yuning-cui/)* and [Jingyuan Xia](https://www.xiajingyuan.com/)

[![paper](https://img.shields.io/badge/IEEE-Paper-00629B.svg)](https://ieeexplore.ieee.org/document/11263975)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<hr />

> **Abstract:** *Transformer-based architectures exhibit substantial promise in the realm of ultra-high-definition (UHD) image restoration (IR). Nevertheless, they encounter significant challenges in maintaining high-frequency (HF) details, which are crucial for the reconstruction of texture. Conventional methods tackle computational complexity by significantly reducing the resolution (by a factor of 4 to 8). Moreover, the majority of high-frequency components are eliminated due to the inherent characteristics of self-attention mechanisms, as these mechanisms tend to naturally suppress high-frequency elements during non-local feature integration. This paper proposes a dual-branch transformer architecture that synergistically combines native-resolution HF preservation with efficient contextual modeling, named HiFormer. The high-resolution branch utilizes a directionally-sensitive large-kernel decomposition to effectively address anisotropic degradations with fewer parameters and applies depthwise separable convolutions for localized high-frequency (HF) information extraction. Concurrently, the low-resolution branch assimilates these localized HF elements using adaptive channel modulation to offset spectral losses induced by the inherent smoothing effect of self-attention. Comprehensive experiments across numerous UHD image restoration tasks reveal that our approach surpasses current leading methods in both quantitative metrics and qualitative analysis.* 

<hr />

## 📋 Table of Contents

- [Model Architecture](#️-model-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Testing](#-testing)
- [Results](#-results)
- [Tips & Tricks](#-tips--tricks)
- [Citation](#-citation)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

## 🏗️ Model Architecture

### HiFormer Overview

<img src = "figs/hiformer_pipeline.png">

**High-Resolution Path:**

- Directionally-decomposed large kernels to efficiently model anisotropic degradations
- Explicit high-frequency mining using depthwise convolutions to extract fine details

**Low-Resolution Path:**

- Self-attention mechanisms for global context understanding
- Adaptive high-frequency compensation that uses details from the high-res path to counteract the spectral losses caused by downsampling and attention's inherent low-pass filtering

### Model Statistics

- **Parameters**: ~2.16M
-  **Inference Memory**: <12G for UHD (4K) Images, even smaller for BF16 precision.

## ✨ Features

### Supported Tasks

| Task | Dataset Code | Dataset | Description |
|------|--------------|---------|-------------|
| **Low-Light Enhancement** | `lol4k` | UHD-LOL4K | Enhance low-light 4K images |
| **Low-Light Enhancement** | `uhd-ll` | UHD-LL | Enhance real-wolrd low-light UHD images |
| **Deraining** | `rain4k` | 4K-Rain13k | Remove rain streaks from 4K images |
| **Deraining** | `uhd-rain` | UHD-Rain | Remove rain streaks from 4K images |
| **Dehazing** | `uhd-haze` | UHD-Haze | Remove haze from UHD images |
| **Deblurring** | `uhd-blur` | UHD-Blur | Remove blur from UHD images |
| **Snow Removal** | `uhd-snow` | UHD-Snow | Remove snow artifacts from UHD images |

- 🚀 **Efficient UHD Processing**: Optimized for 4K and higher resolution images
- 🎨 **Multi-Task Support**: Handles multiple degradation types
- ⚡ **Mixed Precision Training**: Faster training with lower memory usage
- 📊 **Comprehensive Logging**: WandB and TensorBoard integration

## 🔧 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.8.1
- CUDA >= 11.6 (for GPU training)

### Setup

```bash
# Clone the repository
cd HiFormer

# Create conda environment
conda env create -f env.yml
conda activate hiformer
```

## 📄 Project Structure

```
HiFormer/
├── net/
│   └── hiformer.py           # Model definition
├── utils/                    # Utilities
│   ├── dataset_utils.py      # Dataset loaders
│   ├── schedulers.py         # LR schedulers
│   └── ...
├── train_hiformer.py         # Training script
├── test_hiformer.py          # Evaluation script (with metrics)
├── demo_hiformer.py          # Demo script (inference only)
├── options_hiformer.py       # Configuration
├── train_hiformer.sh         # Training launcher
├── test_hiformer.sh          # Testing launcher
└── README.md                 # This file
```

## 📁 Dataset Preparation

### Directory Structure

Organize your datasets as follows:

```
data/
└── Train/
    ├── UHD-haze/
    │   ├── input/    # Hazy images
    │   └── gt/       # Clear images
    ├── UHD-rain/
    │   ├── input/    # Rainy images
    │   └── gt/       # Clean images
    └── ...

data_dir/
├── UHD-haze/
│   └── UHD-haze.txt      # UHD-Haze dataset list (uhd-haze)
├── UHD-rain/
│   └── UHD-rain.txt      # 4K-Rain dataset list (rain4k)
├── LOL-4K/
│   └── UHD_LOL4K.txt     # LOL4K dataset list (lol4k)
├── UHD-LL/
│   └── UHD-LL.txt        # UHD-LL dataset list (uhd-ll)
├── UHD-blur/
│   └── UHD-blur.txt    # UHD-Blur dataset list (uhd-blur)
├── UHD-snow/
│   └── UHD-snow.txt      # UHD-Snow dataset list (uhd-snow)
└── ...
```

### Dataset Lists

Create text files listing your training images:

```bash
# Example: data_dir/hazy/hazy_UHD.txt
data/Train/UHD_haze/train/input/25_250000111.jpg
data/Train/UHD_haze/train/input/37_37000032.jpg
...
```

## 🚀 Training

### Quick Start

```bash
# Train with default settings (UHD dehazing)
bash train_hiformer.sh

# Or run directly
python train_hiformer.py --de_type uhd-haze --epochs 500
```

### Training Options

```bash
python train_hiformer.py \
    --de_type uhd-haze \             # Task type (uhd-haze, uhd-blur, uhd-ll, uhd-snow, lol4k, rain4k)
    --epochs 500 \                   # Training epochs
    --batch_size 8 \                 # Batch size per GPU
    --patch_size 128 \               # Input patch size
    --num_gpus 2 \                   # Number of GPUs
    --lr 2e-4 \                      # Learning rate
    --use_amp \                      # Use mixed precision
    --gradient_clip \                # Enable gradient clipping
    --ckpt_dir ckpt/hiformer/ \     # Checkpoint directory
    --wblogger hiformer              # WandB project name
```

### Resume Training

```bash
python train_hiformer.py \
    --resume_from ckpt/hiformer/hiformer-epoch-100.ckpt \
    --epochs 500
```

## 🧪 Testing

### Evaluation on Test Datasets

For evaluating PSNR/SSIM metrics on test datasets with ground truth:

```bash
# Test on UHD-Haze dataset
python test_hiformer.py \
    --valid_data_dir data/Test/UHD_haze/test/input/ \
    --ckpt_path ckpt/hiformer/hiformer-epoch-499.ckpt \
    --output_path output/

# Test on UHD-Rain dataset
python test_hiformer.py \
    --valid_data_dir data/Test/UHD_rain/test/input/ \
    --ckpt_path ckpt/hiformer/hiformer-epoch-499.ckpt \
    --output_path output/

# Test on LOL4K dataset
python test_hiformer.py \
    --valid_data_dir data/Test/LOL4K/test/input/ \
    --ckpt_path ckpt/hiformer/hiformer-epoch-499.ckpt \
    --output_path output/
```

### Demo: Generate Restored Images

For generating restored images from degraded inputs (without ground truth):

```bash
# Process a directory of images
python demo_hiformer.py \
    --test_path test/input/ \
    --output_path test/output/ \
    --ckpt_path ckpt/hiformer/hiformer-epoch-499.ckpt

# Process with tiling (for very large images)
python demo_hiformer.py \
    --test_path test/input/ \
    --output_path test/output/ \
    --ckpt_path ckpt/hiformer/hiformer-epoch-499.ckpt \
    --tile True \
    --tile_size 512 \
    --tile_overlap 32
```

### Custom Testing

**test_hiformer.py** (for evaluation with metrics):
- `--valid_data_dir`: Path to test input images (requires corresponding GT in `gt/` folder)
- `--ckpt_path`: Path to checkpoint file
- `--output_path`: Directory to save results
- `--cuda`: GPU device ID (default: 0)

**demo_hiformer.py** (for inference only):
- `--test_path`: Path to input images (directory or single image)
- `--output_path`: Directory to save restored images
- `--ckpt_path`: Path to checkpoint file
- `--tile`: Enable tiling for large images (default: False)
- `--tile_size`: Tile size for tiling mode (default: 128)
- `--tile_overlap`: Overlap between tiles (default: 32)
- `--cuda`: GPU device ID (default: 0)

## 📊 Results

### Quantitative Results

<details>
<summary><strong>Low-light UHD Image Enhancement</strong> (click to expand) </summary>
<p align='center'>
<img src = "figs/1.UHDLOL4k.png"> 
</details>

<details>
<summary><strong>UHD Image Deraining</strong> (click to expand) </summary>
<p align='center'>
<img src = "figs/2.UHDRain.png">  
</details>

<details>
<summary><strong>UHD Image Dehazing</strong> (click to expand) </summary>
<p align='center'>
<img src = "figs/3.UHDHaze.png">  
</details>

<details>
<summary><strong>UHD Image Debluring</strong> (click to expand) </summary>
<p align='center'>
<img src = "figs/4.UHDBlur.png">  
</details>

<details>
<summary><strong>UHD Image Desnowing</strong> (click to expand) </summary>
<p align='center'>
<img src = "figs/5.UHDSnow.png">  
</details>

## 💡 Tips & Tricks

### Memory Optimization

If you encounter OOM errors:

```bash
# Reduce batch size and patch size
python train_hiformer.py --batch_size 4 --patch_size 128 --use_amp

# Use gradient accumulation
python train_hiformer.py --batch_size 2 --accumulate_grad_batches 4
```

### Speed Optimization

For faster training:

```bash
# Use more workers
python train_hiformer.py --num_workers 32

# Use multiple GPUs
python train_hiformer.py --num_gpus 4

# Enable mixed precision
python train_hiformer.py --use_amp
```

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{11263975,
  author={Wu, Chen and Wang, Ling and Zheng, Zhuoran and Jiang, Weidong and Cui, Yuning and Xia, Jingyuan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Ultra-High-Definition Image Restoration via High-Frequency Enhanced Transformer}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image restoration;Transformers;MODFETs;HEMTs;High frequency;Degradation;Frequency-domain analysis;Faces;Computational modeling;Videos;Image restoration;UHD image;frequency learning;Transformer},
  doi={10.1109/TCSVT.2025.3636011}
}
```

## 📧 Contact

For any questions or issues, please **open an issue on GitHub**, or contact wuchen5X@mail.ustc.edu.cn, davis0wang@outlook.com.

## 🙏 Acknowledgments

This work builds upon: [PromptIR](https://github.com/va1shn9v/promptir), [Restormer](https://github.com/swz30/Restormer) and [PyTorch Lightning](https://lightning.ai/) repositories. 

## 📝 License

This project is released under the MIT License. See `LICENSE` for details.
