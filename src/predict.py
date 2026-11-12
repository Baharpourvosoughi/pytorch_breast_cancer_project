"""Run a prediction for one feature row using the saved model.

This is a technical demonstration only, not a medical diagnostic tool.
The example input is taken from the built-in dataset for reproducibility.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer

from src.model import BinaryClassifier


def main():
    checkpoint_path = Path("models/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Model not found. Run `python -m src.train` first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BinaryClassifier(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = load_breast_cancer()
    sample = dataset.data[0].astype(np.float32)
    mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
    sample_scaled = (sample - mean) / scale

    with torch.no_grad():
        logits = model(torch.tensor(sample_scaled).view(1, -1))
        probability_class_1 = torch.sigmoid(logits).item()

    print(f"Model probability for class 1 (benign): {probability_class_1:.3f}")
    print("Educational example only; not suitable for medical decisions.")


if __name__ == "__main__":
    main()
