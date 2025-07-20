"""
UBMF: Uncertainty-Aware Bayesian Filtering (Lian et al., 2025)
"""

import torch
import numpy as np

def uncertainty_filter(logits, threshold=0.7):
    """
    Filters predictions with low confidence.
    """
    probs = torch.softmax(logits, dim=1)
    confidences, _ = torch.max(probs, dim=1)
    mask = confidences >= threshold
    return mask
