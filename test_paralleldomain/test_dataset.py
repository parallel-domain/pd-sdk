import pytest

from paralleldomain.dataset import Dataset


def test_can_load_dataset_from_path():
    dataset = Dataset.from_path(dataset_path="s3://paralleldomain-staging/tri-ml/pd_phase2_smallbatch_06_16_2020/")

    assert len(dataset.scenes) > 0