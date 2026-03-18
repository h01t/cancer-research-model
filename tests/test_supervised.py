import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.data.dataset import CBISDDSMDataset
from src.data.transforms import get_transforms
from src.models.efficientnet import EfficientNetClassifier
from src.training.trainer import BaseTrainer


def test_dataset_loading():
    """Test that dataset loads correctly."""
    transform = get_transforms("test", image_size=224)
    dataset = CBISDDSMDataset(
        split="train",
        abnormality_type="mass",
        labeled_subset_size=10,
        transform=transform,
        data_dir="data",
    )
    assert len(dataset) == 10
    img, label = dataset[0]
    assert img.shape == (3, 224, 224)
    assert label in [0, 1]
    print("Dataset test passed")


def test_model_forward():
    """Test model forward pass."""
    model = EfficientNetClassifier(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
    print("Model forward test passed")


def test_trainer_init():
    """Test trainer initialization."""
    config = {
        "training": {
            "batch_size": 4,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "early_stopping_patience": 5,
        },
        "logging": {
            "log_dir": "logs/test",
            "save_dir": "checkpoints/test",
            "save_freq": 1,
            "log_freq": 1,
            "use_tensorboard": False,
        },
    }
    model = EfficientNetClassifier(num_classes=2, pretrained=False)
    trainer = BaseTrainer(model, config)
    assert trainer.device.type in ["cpu", "cuda", "mps"]
    print("Trainer init test passed")


def test_supervised_training_single_batch():
    """Test training loop with a single batch."""
    # Create tiny dataset
    transform = get_transforms("weak", image_size=224)
    dataset = CBISDDSMDataset(
        split="train",
        abnormality_type="mass",
        labeled_subset_size=4,
        transform=transform,
        data_dir="data",
    )
    # Create dataloader with batch size 2
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Model and trainer
    model = EfficientNetClassifier(num_classes=2, pretrained=False)
    config = {
        "training": {
            "batch_size": 2,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adam",
            "scheduler": None,
            "early_stopping_patience": 5,
        },
        "logging": {
            "log_dir": "logs/test",
            "save_dir": "checkpoints/test",
            "save_freq": 1,
            "log_freq": 1,
            "use_tensorboard": False,
        },
    }
    trainer = BaseTrainer(model, config)

    # Train for one batch
    trainer.model.train()
    data, targets = next(iter(loader))
    data, targets = data.to(trainer.device), targets.to(trainer.device)
    trainer.optimizer.zero_grad()
    outputs = trainer.model(data)
    loss = trainer.criterion(outputs, targets)
    loss.backward()
    trainer.optimizer.step()

    # Ensure loss is finite
    assert torch.isfinite(loss)
    print("Single batch training test passed")


if __name__ == "__main__":
    test_dataset_loading()
    test_model_forward()
    test_trainer_init()
    test_supervised_training_single_batch()
    print("All tests passed!")
