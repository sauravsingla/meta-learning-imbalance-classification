"""
REMEDI Ensemble Fusion Module (Liu et al., 2025)
"""

import torch
import numpy as np

def meta_fusion(predictions_list):
    """
    Fuses predictions using learned weights or simple averaging (demo version).
    """
    stacked = np.stack(predictions_list, axis=0)
    return np.mean(stacked, axis=0)
