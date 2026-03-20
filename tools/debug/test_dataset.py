import sys

sys.path.insert(0, ".")

from src.data.dataset import CBISDDSMDataset
from torchvision import transforms

# Test loading training mass dataset
print("Testing dataset loading...")
try:
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    dataset = CBISDDSMDataset(
        split="train",
        abnormality_type="mass",
        labeled_subset_size=10,
        transform=transform,
        data_dir="data",
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Class counts: {dataset.get_class_counts()}")

    # Get first sample
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, label: {label}")

    # Test loading without subset
    dataset_full = CBISDDSMDataset(
        split="train",
        abnormality_type="mass",
        labeled_subset_size=None,
        transform=transform,
        data_dir="data",
    )
    print(f"Full dataset size: {len(dataset_full)}")

    # Test calc dataset
    dataset_calc = CBISDDSMDataset(
        split="train",
        abnormality_type="calc",
        labeled_subset_size=None,
        transform=transform,
        data_dir="data",
    )
    print(f"Calc dataset size: {len(dataset_calc)}")

    # Test both
    dataset_both = CBISDDSMDataset(
        split="train",
        abnormality_type="both",
        labeled_subset_size=None,
        transform=transform,
        data_dir="data",
    )
    print(f"Both dataset size: {len(dataset_both)}")

    print("All tests passed!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
