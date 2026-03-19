"""
Helpers for deterministic, class-balanced labeled subset sampling.
"""

import random


def sample_balanced_labeled_indices(
    candidate_indices: list[int],
    labels: list[int],
    num_labeled: int,
    seed: int,
) -> list[int]:
    """Sample a class-balanced labeled subset from candidate indices."""
    rng = random.Random(seed)
    benign_pool = [idx for idx in candidate_indices if labels[idx] == 0]
    malignant_pool = [idx for idx in candidate_indices if labels[idx] == 1]

    num_per_class = num_labeled // 2
    if len(benign_pool) < num_per_class or len(malignant_pool) < num_per_class:
        raise ValueError(
            f"Not enough samples per class for {num_labeled} labeled. "
            f"Benign: {len(benign_pool)}, Malignant: {len(malignant_pool)}"
        )

    labeled_indices = rng.sample(benign_pool, num_per_class) + rng.sample(
        malignant_pool, num_per_class
    )
    rng.shuffle(labeled_indices)
    return labeled_indices
