import sys
import io
import base64

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from PIL import Image
from scipy.ndimage import zoom

from config import BASE, DATASET_CONFIGS, PRESET_H5, PRESET_SLICES, get_meta

sys.path.insert(0, BASE)
from models.bemunet.bemunet import BEMUNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(name: str) -> BEMUNet:
    cfg = DATASET_CONFIGS[name]
    print(f'[INFO] Loading {name} ...')
    model = BEMUNet(
        num_classes=cfg['num_classes'],
        input_channels=3,
        depths=[2, 2, 2, 2],
        depths_decoder=[2, 2, 2, 1],
        drop_path_rate=0.2,
        load_ckpt_path=None,
    )
    raw = torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=False)
    if isinstance(raw, dict) and 'model' in raw:
        state = raw['model']
    elif isinstance(raw, dict) and 'state_dict' in raw:
        state = raw['state_dict']
    else:
        state = raw

    state = {k: v for k, v in state.items()
             if 'total_ops' not in k and 'total_params' not in k}

    first = next(iter(state))
    if first.startswith('vmunet.'):
        state = {k.replace('vmunet.', 'bemunet.', 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f'[INFO] {name} ready.')
    return model


MODELS: dict[str, BEMUNet] = {name: _load_model(name) for name in DATASET_CONFIGS}
print(f'[INFO] All models loaded on {DEVICE}.')


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray, cfg: dict) -> torch.Tensor:
    th, tw = cfg['input_size']
    if cfg['binary']:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[..., :3]
        img = img.astype(np.float32)
        img = (img - cfg['norm_mean']) / cfg['norm_std']
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / (hi - lo) * 255.0 if hi > lo else img
        h, w = img.shape[:2]
        if (h, w) != (th, tw):
            img = zoom(img, (th / h, tw / w, 1), order=3)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    else:
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / img.max()
        h, w = img.shape
        if (h, w) != (th, tw):
            img = zoom(img, (th / h, tw / w), order=3)
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(tensor: torch.Tensor, dataset: str) -> np.ndarray:
    cfg   = DATASET_CONFIGS[dataset]
    out   = MODELS[dataset](tensor)
    if cfg['binary']:
        return (out.squeeze().cpu().numpy() >= cfg['threshold']).astype(np.uint8)
    pred = torch.argmax(F.softmax(out, dim=1), dim=1)
    return pred.squeeze().cpu().numpy().astype(np.uint8)


# ── Visualisation ─────────────────────────────────────────────────────────────

def colorise(pred: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[pred]


def blend(gray: np.ndarray, pred: np.ndarray,
          palette: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    gray3 = np.stack([gray] * 3, axis=-1).astype(np.float32)
    color = colorise(pred, palette).astype(np.float32) / 255.0
    mask  = pred > 0
    out   = gray3.copy()
    out[mask] = (1 - alpha) * gray3[mask] + alpha * color[mask]
    return (out * 255).clip(0, 255).astype(np.uint8)


def to_b64png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_per_class(pred: np.ndarray, gt: np.ndarray,
                   classes: dict, binary: bool) -> list:
    if gt.shape != pred.shape:
        gt = zoom(gt.astype(np.float32),
                  (pred.shape[0] / gt.shape[0], pred.shape[1] / gt.shape[1]),
                  order=0).astype(np.uint8)
    results = []
    ids = [1] if binary else [c for c in classes if c > 0]
    for cls_id in ids:
        p, g = (pred == cls_id), (gt == cls_id)
        inter = int((p & g).sum())
        denom = int(p.sum()) + int(g.sum())
        results.append({
            'id':     cls_id,
            'name':   classes[cls_id],
            'dice':   round(2 * inter / denom, 4) if denom > 0 else None,
            'has_gt': int(g.sum()) > 0,
        })
    return results


def class_areas(pred: np.ndarray, classes: dict, palette: np.ndarray) -> list:
    fg  = int((pred > 0).sum())
    tot = fg if fg > 0 else 1
    return [
        {
            'id':      c,
            'name':    n,
            'color':   '#{:02x}{:02x}{:02x}'.format(*palette[c]),
            'pixels':  px,
            'percent': round(px / tot * 100, 2),
        }
        for c, n in classes.items()
        if c > 0 and (px := int((pred == c).sum())) > 0
    ]


# ── Preset helpers ────────────────────────────────────────────────────────────

def load_preset(slice_idx: int):
    with h5py.File(PRESET_H5, 'r') as f:
        return f['image'][slice_idx], f['label'][slice_idx]


def preset_thumbnail(img: np.ndarray, size: int = 128) -> str:
    mn, mx = img.min(), img.max()
    u8 = ((img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn \
         else np.zeros_like(img, dtype=np.uint8)
    thumb = zoom(u8.astype(np.float32),
                 (size / u8.shape[0], size / u8.shape[1]), order=1).astype(np.uint8)
    return to_b64png(thumb)
