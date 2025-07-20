# Meta-Learning for Extremely Imbalanced Classification

This project demonstrates a meta-learning-based classification pipeline for extremely imbalanced datasets, inspired by recent research from 2025. It incorporates the following key innovations:

- **IBI3 Boundary-Aware Feature Interpolation (Li et al., 2025)**
- **REMEDI Ensemble Meta-Fusion (Liu et al., 2025)**
- **UBMF Bayesian Filtering + Few-Shot Adaptation (Lian et al., 2025)**
- **IMMAX Margin-Based Meta-Learning (Cortes et al., 2025)**

## 🚀 Highlights

- Meta-learned episodic training loop
- Feature-level interpolation for rare class enhancement
- Model-agnostic meta-learning (MAML-style)
- Bayesian adaptation and ensemble distillation logic
- F1, G-mean, AUC reporting for extreme class imbalance

## 📦 Requirements

```bash
pip install -r requirements.txt
```

## 🧪 Run

```bash
python train_meta.py
```

## 📚 References

- Li et al., Neural Networks, 2025 – IBI3 metric
- Liu et al., REMEDI pipeline, 2025
- Lian et al., UBMF, 2025
- Cortes et al., IMMAX, ICML 2025
- Springer Survey on Meta-Learning for Imbalance, 2024
