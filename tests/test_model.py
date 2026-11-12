import torch
from src.model import BinaryClassifier


def test_model_output_shape():
    model = BinaryClassifier(input_dim=30)
    batch = torch.randn(8, 30)
    output = model(batch)
    assert output.shape == (8, 1)


def test_model_outputs_finite_values():
    model = BinaryClassifier(input_dim=30)
    output = model(torch.randn(4, 30))
    assert torch.isfinite(output).all()
