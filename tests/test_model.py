"""
Unit tests for src/model.py
Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from model import SimpleCNN, get_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
CIFAR_INPUT = (BATCH, 3, 32, 32)
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_resnet18_output_shape(self):
        model = get_model("resnet18", num_classes=NUM_CLASSES)
        x = torch.randn(*CIFAR_INPUT)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES), f"Expected ({BATCH}, {NUM_CLASSES}), got {out.shape}"

    def test_resnet34_output_shape(self):
        model = get_model("resnet34", num_classes=NUM_CLASSES)
        x = torch.randn(*CIFAR_INPUT)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_simple_cnn_output_shape(self):
        model = get_model("simple_cnn", num_classes=NUM_CLASSES)
        x = torch.randn(*CIFAR_INPUT)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_invalid_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model("fakemodel")

    def test_custom_num_classes(self):
        for arch in ["resnet18", "simple_cnn"]:
            model = get_model(arch, num_classes=100)
            x = torch.randn(*CIFAR_INPUT)
            out = model(x)
            assert out.shape == (BATCH, 100), f"Failed for arch={arch}"


class TestSimpleCNN:
    def test_forward_pass(self):
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_eval_mode_no_dropout(self):
        model = SimpleCNN(num_classes=10).eval()
        x = torch.randn(2, 3, 32, 32)
        # Two forward passes should be deterministic in eval mode
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Model not deterministic in eval mode"
