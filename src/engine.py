"""Training, validation, and prediction utilities."""
from copy import deepcopy
from typing import Dict, List

import torch
from torch import nn


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 12,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_probabilities(model, loader, device: str = "cpu"):
    model.eval()
    all_probabilities = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        logits = model(features)
        probabilities = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probabilities.extend(probabilities.tolist())
        all_labels.extend(labels.numpy().ravel().tolist())

    return all_labels, all_probabilities
