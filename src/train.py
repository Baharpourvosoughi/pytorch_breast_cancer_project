"""End-to-end training entry point."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.data import load_data
from src.engine import predict_probabilities, train_model
from src.evaluate import evaluate
from src.model import BinaryClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    # Apple Silicon may use MPS. CPU is a reliable fallback.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    set_seed(42)
    device = choose_device()
    print(f"Using device: {device}")

    bundle = load_data(batch_size=32, random_state=42)
    model = BinaryClassifier(input_dim=bundle.input_dim).to(device)

    history = train_model(
        model,
        bundle.train_loader,
        bundle.val_loader,
        epochs=100,
        learning_rate=1e-3,
        patience=12,
        device=str(device),
    )

    y_true, probabilities = predict_probabilities(
        model, bundle.test_loader, device=str(device)
    )
    metrics = evaluate(y_true, probabilities)

    Path("models").mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_dim": bundle.input_dim,
            "feature_names": bundle.feature_names,
            "scaler_mean": bundle.scaler.mean_.tolist(),
            "scaler_scale": bundle.scaler.scale_.tolist(),
            "threshold": 0.5,
        },
        "models/best_model.pt",
    )

    with open("reports/training_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print("\nTest-set metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    print("\nSaved model to models/best_model.pt")
    print("Saved metrics and plots under reports/")


if __name__ == "__main__":
    main()
