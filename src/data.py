"""Load and split the built-in Breast Cancer Wisconsin dataset."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    feature_names: list
    input_dim: int


def load_data(
    batch_size: int = 32,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> DatasetBundle:
    """Load data, split it, fit scaling on train only, and build loaders."""
    dataset = load_breast_cancer()
    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32)

    # First hold out the test set. It is not used during model selection.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # val_size is the fraction of the remaining train/validation data.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    def make_loader(features, labels, shuffle):
        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        return DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return DatasetBundle(
        train_loader=make_loader(X_train, y_train, True),
        val_loader=make_loader(X_val, y_val, False),
        test_loader=make_loader(X_test, y_test, False),
        scaler=scaler,
        feature_names=list(dataset.feature_names),
        input_dim=X.shape[1],
    )
