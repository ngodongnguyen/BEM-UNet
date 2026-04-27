import io
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import DATASET_CONFIGS, PRESET_SLICES, get_meta
from inference import (
    DEVICE, preprocess, run_inference,
    blend, colorise, to_b64png,
    dice_per_class, class_areas,
    load_preset, preset_thumbnail,
)

app = FastAPI(title='BEM-UNet Demo API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── /api/datasets ─────────────────────────────────────────────────────────────

@app.get('/api/datasets')
def get_datasets():
    return [
        {'key': k, 'label': v['label'], 'metric': v['metric'], 'binary': v['binary']}
        for k, v in DATASET_CONFIGS.items()
    ]


# ── /api/palette ──────────────────────────────────────────────────────────────

@app.get('/api/palette')
def get_palette(dataset: str = 'synapse'):
    if dataset not in DATASET_CONFIGS:
        raise HTTPException(400, f'Unknown dataset: {dataset}')
    _, classes, pal = get_meta(dataset)
    return [
        {'id': c, 'name': n, 'color': '#{:02x}{:02x}{:02x}'.format(*pal[c])}
        for c, n in classes.items() if c > 0
    ]


# ── /api/presets ──────────────────────────────────────────────────────────────

@app.get('/api/presets')
def get_presets():
    result = []
    for idx in PRESET_SLICES:
        img, _ = load_preset(idx)
        result.append({
            'slice': idx,
            'label': f'Slice {idx}',
            'thumb': preset_thumbnail(img),
        })
    return result


# ── /api/predict ──────────────────────────────────────────────────────────────

@app.post('/api/predict')
async def predict(
    dataset:      str            = Form('synapse'),
    preset_slice: Optional[int]  = Form(None),
    slice:        Optional[int]  = Form(None),
    file:         Optional[UploadFile] = File(None),
):
    if dataset not in DATASET_CONFIGS:
        raise HTTPException(400, f'Unknown dataset: {dataset}')

    cfg, classes, palette_arr = get_meta(dataset)
    gt_np     = None
    is_preset = False

    # ── Source image ──────────────────────────────────────────
    if preset_slice is not None:
        img_raw, gt_np = load_preset(preset_slice)
        img_np    = img_raw
        is_preset = True

    elif file is not None:
        raw_bytes = await file.read()
        name = (file.filename or '').lower()

        if name.endswith('.npz'):
            data   = np.load(io.BytesIO(raw_bytes))
            if 'image' not in data:
                raise HTTPException(400, 'NPZ must contain "image" key')
            img_np = data['image']
            gt_np  = data.get('label', None)
            if img_np.ndim == 3:
                s      = slice if slice is not None else img_np.shape[0] // 2
                s      = max(0, min(s, img_np.shape[0] - 1))
                img_np = img_np[s]
                if gt_np is not None:
                    gt_np = gt_np[s]
        elif name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            mode   = 'RGB' if cfg['binary'] else 'L'
            img_np = np.array(Image.open(io.BytesIO(raw_bytes)).convert(mode))
        else:
            raise HTTPException(400, 'Unsupported file type. Use PNG/JPG/NPZ.')
    else:
        raise HTTPException(400, 'Provide file or preset_slice')

    # ── Normalise original for display ────────────────────────
    orig_color = None
    if img_np.ndim == 3:
        orig_color = img_np.astype(np.uint8) if img_np.dtype != np.uint8 else img_np
        orig_gray  = np.array(
            Image.fromarray(orig_color).convert('L')
        ).astype(np.float32) / 255.0
    else:
        f = img_np.astype(np.float32)
        orig_gray = f / f.max() if f.max() > 1.0 else f

    oh, ow = orig_gray.shape

    # ── Inference ──────────────────────────────────────────────
    tensor = preprocess(img_np, cfg)
    pred   = run_inference(tensor, dataset)

    ih, iw = cfg['input_size']
    if (oh, ow) != (ih, iw):
        from scipy.ndimage import zoom as _zoom
        pred = _zoom(pred.astype(np.float32), (oh / ih, ow / iw), order=0).astype(np.uint8)

    # ── Build images ───────────────────────────────────────────
    orig_u8 = orig_color if orig_color is not None \
              else (orig_gray * 255).clip(0, 255).astype(np.uint8)
    mask_b64 = to_b64png((pred * 255).astype(np.uint8)) if cfg['binary'] else None

    # ── GT overlay + metrics ───────────────────────────────────
    gt_b64  = None
    metrics = None
    if gt_np is not None:
        from scipy.ndimage import zoom as _zoom
        gt_r = gt_np.astype(np.uint8)
        if gt_r.shape != (oh, ow):
            gt_r = _zoom(gt_r.astype(np.float32),
                         (oh / gt_r.shape[0], ow / gt_r.shape[1]),
                         order=0).astype(np.uint8)
        gt_b64     = to_b64png(blend(orig_gray, gt_r, palette_arr))
        per_class  = dice_per_class(pred, gt_r, classes, cfg['binary'])
        for c in per_class:
            c['color'] = '#{:02x}{:02x}{:02x}'.format(*palette_arr[c['id']])
        valid      = [c['dice'] for c in per_class if c['dice'] is not None and c['has_gt']]
        metrics    = {'per_class': per_class,
                      'mean_dice': round(sum(valid) / len(valid), 4) if valid else None}

    return {
        'original':     to_b64png(orig_u8),
        'segmentation': to_b64png(colorise(pred, palette_arr)),
        'overlay':      to_b64png(blend(orig_gray, pred, palette_arr)),
        'mask':         mask_b64,
        'gt_overlay':   gt_b64,
        'classes':      class_areas(pred, classes, palette_arr),
        'metrics':      metrics,
        'image_size':   [oh, ow],
        'dataset':      dataset,
        'is_preset':    is_preset,
        'device':       str(DEVICE),
    }
