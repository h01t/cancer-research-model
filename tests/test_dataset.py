"""
Tests for dataset utilities (patient-aware splitting, class-balanced sampling).

All tests use mock data — no CBIS-DDSM dataset required.
"""

import pytest
import pandas as pd
from torch.utils.data import Subset

from src.data.dataset import extract_metadata_frame, patient_aware_split


class MockDataset:
    """Minimal mock of CBISDDSMDataset with patient_ids and labels."""

    def __init__(self, patient_ids: list[str], labels: list[int]):
        assert len(patient_ids) == len(labels)
        self.patient_ids = patient_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)


class TestPatientAwareSplit:
    """Test patient-aware train/val splitting."""

    def _make_dataset(self):
        """Create a mock dataset with 6 patients, 12 images, mixed labels."""
        # Patient A: 3 images, all benign
        # Patient B: 2 images, all malignant
        # Patient C: 1 image, benign
        # Patient D: 2 images, all malignant
        # Patient E: 2 images, all benign
        # Patient F: 2 images, 1 benign + 1 malignant (mixed → majority benign)
        patient_ids = [
            "A",
            "A",
            "A",  # benign patient
            "B",
            "B",  # malignant patient
            "C",  # benign patient
            "D",
            "D",  # malignant patient
            "E",
            "E",  # benign patient
            "F",
            "F",  # mixed patient (majority benign)
        ]
        labels = [
            0,
            0,
            0,  # A
            1,
            1,  # B
            0,  # C
            1,
            1,  # D
            0,
            0,  # E
            0,
            1,  # F
        ]
        return MockDataset(patient_ids, labels)

    def test_no_patient_overlap(self):
        """No patient should appear in both train and val sets."""
        ds = self._make_dataset()
        train_idx, val_idx = patient_aware_split(ds, val_fraction=0.3, seed=42)

        train_patients = {ds.patient_ids[i] for i in train_idx}
        val_patients = {ds.patient_ids[i] for i in val_idx}

        assert train_patients.isdisjoint(val_patients), (
            f"Patient overlap detected: {train_patients & val_patients}"
        )

    def test_all_indices_covered(self):
        """Every image index must appear in exactly one set."""
        ds = self._make_dataset()
        train_idx, val_idx = patient_aware_split(ds, val_fraction=0.3, seed=42)

        all_idx = sorted(train_idx + val_idx)
        assert all_idx == list(range(len(ds)))

    def test_no_duplicate_indices(self):
        """No index should appear twice."""
        ds = self._make_dataset()
        train_idx, val_idx = patient_aware_split(ds, val_fraction=0.3, seed=42)

        assert len(set(train_idx)) == len(train_idx)
        assert len(set(val_idx)) == len(val_idx)

    def test_val_fraction_respected(self):
        """Val set should contain approximately val_fraction of patients."""
        ds = self._make_dataset()
        train_idx, val_idx = patient_aware_split(ds, val_fraction=0.3, seed=42)

        total_patients = len(set(ds.patient_ids))
        val_patients = len({ds.patient_ids[i] for i in val_idx})

        # With 6 patients and 0.3, expect ~2 val patients (0.3*4 benign ≈1, 0.3*2 mal ≈1)
        assert 1 <= val_patients <= 3

    def test_deterministic_with_same_seed(self):
        """Same seed produces same split."""
        ds = self._make_dataset()
        train1, val1 = patient_aware_split(ds, val_fraction=0.3, seed=42)
        train2, val2 = patient_aware_split(ds, val_fraction=0.3, seed=42)

        assert train1 == train2
        assert val1 == val2

    def test_different_seed_different_split(self):
        """Different seeds produce different splits."""
        ds = self._make_dataset()
        train1, val1 = patient_aware_split(ds, val_fraction=0.3, seed=42)
        train2, val2 = patient_aware_split(ds, val_fraction=0.3, seed=99)

        # With small data, it's possible to get the same split by chance,
        # but extremely unlikely
        val_patients_1 = {ds.patient_ids[i] for i in val1}
        val_patients_2 = {ds.patient_ids[i] for i in val2}
        # At minimum the ordering should differ if patients are the same
        assert val1 != val2 or train1 != train2

    def test_both_classes_in_val(self):
        """Val set should contain both benign and malignant patients."""
        ds = self._make_dataset()
        train_idx, val_idx = patient_aware_split(ds, val_fraction=0.3, seed=42)

        val_labels = {ds.labels[i] for i in val_idx}
        assert 0 in val_labels, "Val set has no benign samples"
        assert 1 in val_labels, "Val set has no malignant samples"

    def test_single_patient_per_class(self):
        """Edge case: only 1 patient per class. Val should still get at least 1."""
        ds = MockDataset(
            patient_ids=["A", "A", "B", "B"],
            labels=[0, 0, 1, 1],
        )
        train_idx, val_idx = patient_aware_split(ds, val_fraction=0.5, seed=42)

        # All indices covered, no overlap
        assert sorted(train_idx + val_idx) == [0, 1, 2, 3]
        train_patients = {ds.patient_ids[i] for i in train_idx}
        val_patients = {ds.patient_ids[i] for i in val_idx}
        assert train_patients.isdisjoint(val_patients)


class MockMetadataDataset:
    def __init__(self):
        self.frame = pd.DataFrame(
            {
                "patient_id": ["A", "A", "B"],
                "exam_id": ["A_L", "A_R", "B_L"],
                "label": [0, 1, 0],
            }
        )

    def get_metadata_frame(self):
        return self.frame.copy()


class TestMetadataExtraction:
    def test_extract_metadata_from_subset(self):
        dataset = MockMetadataDataset()
        subset = Subset(dataset, [2, 0])
        frame = extract_metadata_frame(subset)

        assert list(frame["patient_id"]) == ["B", "A"]
