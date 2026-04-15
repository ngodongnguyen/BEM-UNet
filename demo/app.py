import sys
import os
import io
import base64
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from flask import Flask, request, jsonify, render_template

from models.bemunet.bemunet import BEMUNet

# ──────────────────────────────────────────────
# Dataset configs
# ──────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE   = os.path.join(os.path.dirname(__file__), '..')

DATASET_CONFIGS = {
    'synapse': {
        'ckpt':        os.path.join(BASE, 'SYNAPSE', 'best-epoch263-mean_dice0.8493-mean_hd9512.2378.pth'),
        'num_classes': 9,
        'input_size':  (224, 224),
        'binary':      False,
        'threshold':   None,
        # Synapse CT: normalise to [0,1] (handled in preprocess)
        'norm_mean':   None,
        'norm_std':    None,
        'label':       'Synapse — Multi-organ',
        'metric':      'Mean Dice 0.8493',
    },
    'isic18': {
        'ckpt':        os.path.join(BASE, 'ISIC2018', 'best-epoch212-miou0.8142.pth'),
        'num_classes': 1,
        'input_size':  (256, 256),
        'binary':      True,
        'threshold':   0.5,
        # test-split normalisation from utils.py / myNormalize
        'norm_mean':   149.034,
        'norm_std':    32.022,
        'label':       'ISIC 2018 — Skin Lesion',
        'metric':      'mIoU 0.8142',
    },
    'isic17': {
        'ckpt':        os.path.join(BASE, 'ISIC2017', 'best-epoch39-miou0.8065.pth'),
        'num_classes': 1,
        'input_size':  (256, 256),
        'binary':      True,
        'threshold':   0.5,
        'norm_mean':   148.429,
        'norm_std':    25.748,
        'label':       'ISIC 2017 — Skin Lesion',
        'metric':      'mIoU 0.8065',
    },
}

# Synapse 9-class palette (index = class id)
SYNAPSE_CLASSES = {
    0: 'Background',
    1: 'Spleen',
    2: 'Right Kidney',
    3: 'Left Kidney',
    4: 'Gallbladder',
    5: 'Liver',
    6: 'Stomach',
    7: 'Pancreas',
    8: 'Aorta',
}
SYNAPSE_PALETTE = np.array([
    [  0,   0,   0],   # 0 Background
    [220,  50,  50],   # 1 Spleen
    [ 50,  50, 220],   # 2 R-Kidney
    [ 50, 200, 200],   # 3 L-Kidney
    [230, 220,  50],   # 4 Gallbladder
    [ 50, 160,  50],   # 5 Liver
    [230, 130,  30],   # 6 Stomach
    [170,  50, 200],   # 7 Pancreas
    [240, 100, 170],   # 8 Aorta
], dtype=np.uint8)

# ISIC binary palette
ISIC_CLASSES  = {0: 'Background', 1: 'Lesion'}
ISIC_PALETTE  = np.array([[0, 0, 0], [50, 220, 100]], dtype=np.uint8)

# ──────────────────────────────────────────────
# Load all models at startup
# ──────────────────────────────────────────────
MODELS = {}

def _load_model(name: str):
    cfg = DATASET_CONFIGS[name]
    print(f"[INFO] Loading {name} model ...")
    m = BEMUNet(
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
    # strip thop profiling artifacts (total_ops / total_params keys)
    state = {k: v for k, v in state.items()
             if 'total_ops' not in k and 'total_params' not in k}

    # remap old prefix
    first_key = next(iter(state))
    if first_key.startswith('vmunet.'):
        state = {k.replace('vmunet.', 'bemunet.', 1): v for k, v in state.items()}

    m.load_state_dict(state)
    m.to(DEVICE).eval()
    print(f"[INFO] {name} ready.")
    return m

for _name in DATASET_CONFIGS:
    MODELS[_name] = _load_model(_name)

print(f"[INFO] All models loaded on {DEVICE}.")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_meta(dataset: str):
    cfg = DATASET_CONFIGS[dataset]
    if dataset == 'synapse':
        return cfg, SYNAPSE_CLASSES, SYNAPSE_PALETTE
    else:
        return cfg, ISIC_CLASSES, ISIC_PALETTE


def preprocess(img_array: np.ndarray, cfg: dict) -> torch.Tensor:
    """
    Synapse : (H,W) grayscale float/uint8  → (1,1,H,W) normalised to [0,1]
    ISIC    : (H,W,3) or (H,W) RGB uint8  → (1,3,H,W) normalised with dataset mean/std
    """
    target_h, target_w = cfg['input_size']

    if cfg['binary']:
        # ── ISIC: expect RGB ──────────────────────────────────────
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)      # (H,W) → (H,W,3)
        elif img_array.shape[2] == 4:
            img_array = img_array[..., :3]                       # drop alpha

        img_f = img_array.astype(np.float32)                     # [0,255]

        # resize
        h, w = img_f.shape[:2]
        if (h, w) != (target_h, target_w):
            img_f = zoom(img_f, (target_h / h, target_w / w, 1), order=3)

        # normalise: (img - mean) / std  (same as myNormalize, inference split)
        img_f = (img_f - cfg['norm_mean']) / cfg['norm_std']

        # (H,W,3) → (1,3,H,W)
        tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).float().unsqueeze(0)
    else:
        # ── Synapse: grayscale CT ─────────────────────────────────
        if img_array.ndim == 3:
            img_array = img_array[..., 0]                        # take first channel
        img_f = img_array.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / img_f.max()

        h, w = img_f.shape
        if (h, w) != (target_h, target_w):
            img_f = zoom(img_f, (target_h / h, target_w / w), order=3)

        # (H,W) → (1,1,H,W)  ; model repeats 1-ch → 3-ch internally
        tensor = torch.from_numpy(img_f).float().unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


@torch.no_grad()
def run_inference(tensor: torch.Tensor, dataset: str) -> np.ndarray:
    cfg = DATASET_CONFIGS[dataset]
    model = MODELS[dataset]
    out   = model(tensor)

    if cfg['binary']:
        # out is already sigmoid-activated (BEMUNet.forward)
        prob = out.squeeze().cpu().numpy()          # (H,W) float [0,1]
        pred = (prob >= cfg['threshold']).astype(np.uint8)
    else:
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    return pred


def colorise(pred: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[pred]


def blend_overlay(orig_gray: np.ndarray, pred: np.ndarray,
                  palette: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    orig_gray : (H,W) float [0,1]
    pred      : (H,W) uint8 class indices
    """
    gray3  = np.stack([orig_gray] * 3, axis=-1).astype(np.float32)
    color  = colorise(pred, palette).astype(np.float32) / 255.0
    mask   = pred > 0
    out    = gray3.copy()
    out[mask] = (1 - alpha) * gray3[mask] + alpha * color[mask]
    return (out * 255).clip(0, 255).astype(np.uint8)


def ndarray_to_b64png(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def dice_per_class(pred: np.ndarray, gt: np.ndarray,
                   classes: dict, binary: bool) -> list:
    """
    Compute Dice score for each foreground class.
    pred, gt : (H, W) uint8 — must be same shape.
    Returns list of {id, name, color, dice, has_gt}
    """
    # Align gt shape to pred if needed
    if gt.shape != pred.shape:
        gt = zoom(gt.astype(np.float32),
                  (pred.shape[0] / gt.shape[0], pred.shape[1] / gt.shape[1]),
                  order=0).astype(np.uint8)

    results = []
    if binary:
        # single foreground class
        p = (pred == 1)
        g = (gt   == 1)
        inter = int((p & g).sum())
        denom = int(p.sum()) + int(g.sum())
        dice  = round(2 * inter / denom, 4) if denom > 0 else None
        results.append({
            'id':     1,
            'name':   'Lesion',
            'dice':   dice,
            'has_gt': int(g.sum()) > 0,
        })
    else:
        for cls_id, cls_name in classes.items():
            if cls_id == 0:
                continue
            p = (pred == cls_id)
            g = (gt   == cls_id)
            inter = int((p & g).sum())
            denom = int(p.sum()) + int(g.sum())
            has_gt = int(g.sum()) > 0
            dice   = round(2 * inter / denom, 4) if denom > 0 else None
            results.append({
                'id':     cls_id,
                'name':   cls_name,
                'dice':   dice,
                'has_gt': has_gt,
            })
    return results


def compute_class_areas(pred: np.ndarray, classes: dict, palette: np.ndarray):
    fg  = int((pred > 0).sum())
    tot = fg if fg > 0 else 1
    result = []
    for cls_id, cls_name in classes.items():
        if cls_id == 0:
            continue
        px = int((pred == cls_id).sum())
        if px > 0:
            result.append({
                'id':      cls_id,
                'name':    cls_name,
                'color':   '#{:02x}{:02x}{:02x}'.format(*palette[cls_id]),
                'pixels':  px,
                'percent': round(px / tot * 100, 2),
            })
    return result


# ──────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/datasets')
def datasets():
    """Return dataset metadata for the frontend."""
    result = []
    for key, cfg in DATASET_CONFIGS.items():
        result.append({
            'key':    key,
            'label':  cfg['label'],
            'metric': cfg['metric'],
            'binary': cfg['binary'],
        })
    return jsonify(result)


@app.route('/palette')
def palette():
    dataset = request.args.get('dataset', 'synapse')
    if dataset not in DATASET_CONFIGS:
        return jsonify({'error': 'Unknown dataset'}), 400
    _, classes, pal = get_meta(dataset)
    data = []
    for cls_id, cls_name in classes.items():
        if cls_id == 0:
            continue
        data.append({
            'id':    cls_id,
            'name':  cls_name,
            'color': '#{:02x}{:02x}{:02x}'.format(*pal[cls_id]),
        })
    return jsonify(data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file    = request.files['file']
        dataset = request.form.get('dataset', 'synapse')
        if dataset not in DATASET_CONFIGS:
            return jsonify({'error': f'Unknown dataset: {dataset}'}), 400

        cfg, classes, palette_arr = get_meta(dataset)
        filename = file.filename.lower()

        # ── Load image ──────────────────────────────────────────
        gt_np = None
        if filename.endswith('.npz'):
            data = np.load(io.BytesIO(file.read()))
            if 'image' not in data:
                return jsonify({'error': 'NPZ must contain "image" key'}), 400
            img_np = data['image']
            gt_np  = data.get('label', None)
            if img_np.ndim == 3:
                slice_idx = int(request.form.get('slice', img_np.shape[0] // 2))
                slice_idx = max(0, min(slice_idx, img_np.shape[0] - 1))
                img_np = img_np[slice_idx]
                if gt_np is not None:
                    gt_np = gt_np[slice_idx]

        elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            if cfg['binary']:
                # ISIC: keep RGB
                pil_img = Image.open(io.BytesIO(file.read())).convert('RGB')
                img_np  = np.array(pil_img)            # (H,W,3) uint8
            else:
                # Synapse: grayscale
                pil_img = Image.open(io.BytesIO(file.read())).convert('L')
                img_np  = np.array(pil_img)            # (H,W) uint8
        else:
            return jsonify({'error': 'Unsupported file type. Use PNG/JPG/NPZ.'}), 400

        # ── Original image for display (always grayscale float) ─
        if img_np.ndim == 3:
            orig_gray = np.array(Image.fromarray(img_np).convert('L')).astype(np.float32) / 255.0
        else:
            orig_float = img_np.astype(np.float32)
            orig_gray  = orig_float / orig_float.max() if orig_float.max() > 1.0 else orig_float

        oh, ow = orig_gray.shape[:2]

        # ── Inference ────────────────────────────────────────────
        tensor    = preprocess(img_np, cfg)
        pred      = run_inference(tensor, dataset)          # (input_h, input_w)

        # Resize prediction back to original size
        ih, iw = cfg['input_size']
        if (oh, ow) != (ih, iw):
            pred = zoom(pred.astype(np.float32),
                        (oh / ih, ow / iw), order=0).astype(np.uint8)

        # ── Build output images ──────────────────────────────────
        orig_uint8  = (orig_gray * 255).clip(0, 255).astype(np.uint8)
        seg_rgb     = colorise(pred, palette_arr)
        overlay_rgb = blend_overlay(orig_gray, pred, palette_arr)

        # ── GT overlay + metrics (NPZ only) ─────────────────────
        gt_b64  = None
        metrics = None
        if gt_np is not None:
            gt_r = gt_np.astype(np.uint8)
            if gt_r.shape != orig_uint8.shape:
                gt_r = zoom(gt_r.astype(np.float32),
                            (oh / gt_r.shape[0], ow / gt_r.shape[1]),
                            order=0).astype(np.uint8)
            gt_b64 = ndarray_to_b64png(blend_overlay(orig_gray, gt_r, palette_arr))

            # Dice per class
            per_class = dice_per_class(pred, gt_r, classes, cfg['binary'])
            valid = [c['dice'] for c in per_class if c['dice'] is not None and c['has_gt']]
            mean_dice = round(sum(valid) / len(valid), 4) if valid else None

            # Attach colour to each entry
            for c in per_class:
                c['color'] = '#{:02x}{:02x}{:02x}'.format(*palette_arr[c['id']])

            metrics = {'per_class': per_class, 'mean_dice': mean_dice}

        class_info = compute_class_areas(pred, classes, palette_arr)

        return jsonify({
            'original':     ndarray_to_b64png(orig_uint8),
            'segmentation': ndarray_to_b64png(seg_rgb),
            'overlay':      ndarray_to_b64png(overlay_rgb),
            'gt_overlay':   gt_b64,
            'classes':      class_info,
            'metrics':      metrics,
            'image_size':   [oh, ow],
            'dataset':      dataset,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
