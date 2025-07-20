"""
IBI3-style Boundary-Aware Interpolation Module (Li et al., 2025)
"""

import torch

def interpolate_features(X_pos, X_neg, alpha=0.5):
    """
    Generates synthetic boundary points by interpolating between positive and negative examples.
    """
    synthetic = []
    for i in range(len(X_pos)):
        idx = torch.randint(0, len(X_neg), (1,))
        mix = alpha * X_pos[i] + (1 - alpha) * X_neg[idx]
        synthetic.append(mix)
    return torch.stack(synthetic)
