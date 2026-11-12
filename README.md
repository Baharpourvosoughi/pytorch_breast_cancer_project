# Breast Cancer Classification Using PyTorch

An end-to-end, beginner-to-intermediate PyTorch project using the small
Breast Cancer Wisconsin dataset bundled with scikit-learn. No manual dataset
download is required.

> **Important:** This is an educational machine-learning exercise. It is not a
> medical device, diagnostic system, or a basis for health decisions.

## Learning objectives

- Load and inspect a public tabular dataset.
- Split data into training, validation, and test sets using stratification.
- Avoid preprocessing leakage by fitting `StandardScaler` on training data only.
- Build a feed-forward neural network with `torch.nn`.
- Understand logits, `BCEWithLogitsLoss`, gradients, backpropagation, and Adam.
- Train with a validation set and early stopping.
- Evaluate using accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC.
- Save a model checkpoint and create evaluation plots.

## Dataset

The project uses `sklearn.datasets.load_breast_cancer`, a 569-row,
30-feature binary classification dataset. The target labels are encoded as:
- `0`: malignant
- `1`: benign

The data are supplied through scikit-learn, so there is no separate large
archive to download.

## Project structure

```text
pytorch-breast-cancer-project/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── models/
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup on macOS

Open Terminal and move into the project folder:

```bash
cd pytorch-breast-cancer-project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, use the official PyTorch installation selector
for the current macOS/Python combination:
https://pytorch.org/get-started/locally/

## Run the project

From the project root:

```bash
python -m src.train
```

The script:
1. Loads the dataset.
2. Creates train, validation, and test splits.
3. Fits preprocessing on the training split only.
4. Trains a small neural network.
5. Selects the best checkpoint using validation loss.
6. Evaluates once on the held-out test set.
7. Saves the model and reports.

Expected generated files:

```text
models/best_model.pt
reports/metrics.json
reports/training_history.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
```

Metric values can vary slightly across hardware and software versions.
Do not copy expected or guessed scores into a report; use the metrics produced
by your own run.

## Run one demonstration prediction

After training:

```bash
python -m src.predict
```

This uses one example row from the bundled dataset to demonstrate loading the
saved model and applying the stored scaling parameters.

## Run tests

```bash
pytest -q
```

## Core concepts explained

### 1. Tensor
A PyTorch tensor is a multi-dimensional array. Model inputs and outputs are
represented as tensors.

### 2. Neural network
`BinaryClassifier` is a feed-forward network with two hidden layers. Each
`Linear` layer learns weights and biases. ReLU introduces non-linearity and
Dropout regularizes the model during training.

### 3. Logits and loss
The model returns a raw score (logit), not a probability. `BCEWithLogitsLoss`
combines sigmoid and binary cross-entropy in a numerically stable operation.

### 4. Backpropagation
`loss.backward()` computes gradients for trainable parameters. The Adam
optimizer uses those gradients to update the parameters with `optimizer.step()`.

### 5. Validation and test sets
The validation set is used for early stopping/model selection. The test set is
held aside until final evaluation. The scaler is fitted only on training data,
which helps prevent data leakage.

### 6. Metrics
- Accuracy: fraction of all predictions that are correct.
- Balanced accuracy: average recall across the two classes.
- Precision: fraction of predicted positives that are correct.
- Recall: fraction of actual positives identified.
- F1: harmonic mean of precision and recall.
- ROC-AUC: ranking performance across classification thresholds.

The positive class in this dataset is class `1` (benign). Therefore, precision
and recall refer to the benign class. For a serious medical application, class
definitions, error costs, threshold selection, calibration, and external
validation would require careful domain review.

## Limitations

- This is a small, historical, tabular benchmark dataset.
- A random split does not establish generalization to other hospitals,
  populations, or future clinical settings.
- The model is not clinically validated.
- No causal conclusions can be drawn.
- A neural network is not guaranteed to outperform simpler models on this
  dataset. A strong research extension would compare it with logistic
  regression, random forest, and gradient boosting using the same splits.

## Suggested extensions

1. Add a logistic-regression baseline.
2. Compare the neural network with a random forest.
3. Plot training and validation loss curves.
4. Add repeated stratified cross-validation for model comparison.
5. Investigate class-specific precision/recall and threshold trade-offs.
6. Add a short report describing methodology, results, limitations, and
   reproducibility details.

## Reproducibility

The random seed is set to 42 for Python, NumPy, and PyTorch. Exact numerical
reproducibility across different hardware and library versions is not
guaranteed.
