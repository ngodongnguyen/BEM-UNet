# BEM-UNet

Code repository for **BEM-UNet**: a boundary-enhanced medical image segmentation architecture built upon efficient state-space modeling.

## Abstract

Medical image segmentation requires precise boundary delineation while effectively modeling long-range contextual dependencies. Convolutional neural networks (CNNs) excel at capturing local details but struggle with global context, whereas Transformer-based methods introduce high computational complexity. Recently, State Space Models (SSMs) have emerged as an efficient alternative for long-range dependency modeling; however, existing SSM-based segmentation frameworks still suffer from inadequate boundary preservation and limited multi-scale feature interaction. In this paper, we propose **BEM-UNet**, a boundary-enhanced segmentation architecture built upon efficient state-space modeling. The proposed framework integrates **Efficient Vision State Space (E-VSS)** blocks for global context modeling, a **Lightweight Dual-Domain Feature Block** to enhance early representations using both spatial and frequency-domain information, and a **Long–Short SSM bottleneck** to capture multi-scale dependencies. To further improve feature interaction and prediction quality, we introduce **Semantic-Difference Gated Skip Connections** for adaptive encoder–decoder fusion and a **Reverse-Guided Refinement Block** for progressive error correction. Extensive experiments on the ISIC2017, ISIC2018, and Synapse datasets demonstrate that BEM-UNet consistently outperforms existing methods. In particular, the proposed model achieves superior Dice Similarity Coefficient (DSC) and significantly reduces the 95% Hausdorff Distance (HD95), indicating improved segmentation accuracy and boundary delineation.

## Architecture

![BEM-UNet Architecture](architecture.png)

---

## 0. Environment Setup

```bash
conda create -n bemunet python=3.10
conda activate bemunet
```

**Install PyTorch:**
```bash
pip install torch==2.9.0 torchvision==0.24.0
```

**Install dependencies:**
```bash
pip install transformers==4.37.2 timm
pip install h5py scipy tensorboardX scikit-learn matplotlib
pip install SimpleITK medpy thop yacs ml_collections einops
```

**Install `causal-conv1d`, `mamba`, and `selective_scan` from source:**

```bash
git clone https://github.com/Dao-AILab/causal-conv1d
cd causal-conv1d
pip install --no-build-isolation .

git clone https://github.com/state-spaces/mamba
cd mamba
pip install --no-build-isolation -e .

git clone https://github.com/MzeroMiko/VMamba
cd VMamba/kernels/selective_scan
pip install --no-build-isolation .
```

---

## 1. Prepare the Dataset

### ISIC Datasets (Binary Skin Lesion Segmentation)

Download ISIC17 and ISIC18 (split 7:3) and place them as follows:

```
./data/isic2017/
    train/
        images/   *.png
        masks/    *.png
    val/
        images/   *.png
        masks/    *.png

./data/isic2018/
    train/
        images/   *.png
        masks/    *.png
    val/
        images/   *.png
        masks/    *.png
```

### Synapse Dataset (Multi-organ Segmentation)

Follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the Synapse dataset and place it as:

```
./data/Synapse/
    lists/
        list_Synapse/
            all.lst
            test_vol.txt
            train.txt
    test_vol_h5/
        casexxxx.npy.h5
    train_npz/
        casexxxx_slicexxx.npz
```

---

## 2. Pre-trained Weights

Download the VMamba Small pre-trained weights (`vmamba_small_e238_ema.pth`) from [VMamba](https://github.com/MzeroMiko/VMamba) and place them in:

```
./pre_trained_weights/vmamba_small_e238_ema.pth
```

---

## 3. Train BEM-UNet

```bash
# Train on ISIC17 or ISIC18 (set `datasets` in configs/config_setting.py)
python train.py

# Train on Synapse
python train_synapse.py
```

#### Training Hyperparameters

| Setting | ISIC17/18 | Synapse |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 0.001 | 0.001 |
| Scheduler | CosineAnnealingLR (T_max=50) | CosineAnnealingLR |
| Epochs | 300 | 300 |
| Batch size | 1 | 28 |
| Loss | BceDiceLoss (wb=1, wd=1) | CeDiceLoss (CE×0.4 + Dice×0.6) |
| Input size | 256 × 256 | 224 × 224 |

---

## 4. Best Checkpoints

The best trained checkpoints for each dataset are available on Google Drive:

[Download best checkpoints (Google Drive)](https://drive.google.com/drive/folders/1RAlOlFMRWWanGawP_baf2BPZWH2s8yKH)

To run inference with a downloaded checkpoint, set in `configs/config_setting.py`:
```python
only_test_and_save_figs = True
best_ckpt_path = 'PATH_TO_DOWNLOADED_CKPT'
img_save_path  = 'PATH_TO_SAVE_IMAGES'
```
Then run `python train.py`.

---

## 5. Acknowledgments

- [VMamba](https://github.com/MzeroMiko/VMamba) — backbone and pre-trained weights
- [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) — dataset preparation reference
- [VM-UNet](https://arxiv.org/abs/2402.02491) — foundational SSM-based segmentation baseline
