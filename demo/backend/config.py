import os
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATASET_CONFIGS: dict = {
    'synapse': {
        'ckpt':        os.path.join(BASE, 'SYNAPSE', 'best-epoch263-mean_dice0.8493-mean_hd9512.2378.pth'),
        'num_classes': 9,
        'input_size':  (224, 224),
        'binary':      False,
        'threshold':   None,
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

SYNAPSE_CLASSES = {
    0: 'Background', 1: 'Spleen',    2: 'Right Kidney', 3: 'Left Kidney',
    4: 'Gallbladder', 5: 'Liver',    6: 'Stomach',      7: 'Pancreas',
    8: 'Aorta',
}
SYNAPSE_PALETTE = np.array([
    [  0,   0,   0],
    [220,  50,  50],
    [ 50,  50, 220],
    [ 50, 200, 200],
    [230, 220,  50],
    [ 50, 160,  50],
    [230, 130,  30],
    [170,  50, 200],
    [240, 100, 170],
], dtype=np.uint8)

ISIC_CLASSES  = {0: 'Background', 1: 'Lesion'}
ISIC_PALETTE  = np.array([[0, 0, 0], [50, 220, 100]], dtype=np.uint8)

PRESET_H5     = os.path.join(BASE, 'data', 'case0001.npy.h5')
PRESET_SLICES = [100, 110, 120]


def get_meta(dataset: str):
    cfg = DATASET_CONFIGS[dataset]
    if dataset == 'synapse':
        return cfg, SYNAPSE_CLASSES, SYNAPSE_PALETTE
    return cfg, ISIC_CLASSES, ISIC_PALETTE
