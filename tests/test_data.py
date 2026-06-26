"""Tests for synthetic data generation, Dataset, DataLoaders, and transforms."""
import numpy as np
import pytest
import torch

from src.data import (
    H,
    W,
    PathSegDataset,
    create_dataloaders,
    create_datasets,
    make_synthetic,
    prepare_data,
)
from src.transforms import get_transform, train_transform, val_transform


def test_make_synthetic_shapes():
    data = make_synthetic(n=10, seed=42)
    assert data["images"].shape == (10, H, W, 3)
    assert data["masks"].shape == (10, H, W)
    assert data["images"].dtype == np.uint8
    assert data["masks"].dtype == np.uint8


def test_make_synthetic_pixel_range():
    data = make_synthetic(n=5, seed=7)
    assert data["images"].min() >= 0
    assert data["images"].max() <= 255
    assert set(np.unique(data["masks"])).issubset({0, 1})


def test_make_synthetic_has_masks():
    data = make_synthetic(n=20, seed=1)
    positive_rate = data["masks"].mean()
    assert 0.01 < positive_rate < 0.5


def test_make_symmetric_deterministic():
    d1 = make_synthetic(n=5, seed=99)
    d2 = make_synthetic(n=5, seed=99)
    assert np.array_equal(d1["images"], d2["images"])
    assert np.array_equal(d1["masks"], d2["masks"])


def test_make_synthetic_different_seeds():
    d1 = make_synthetic(n=5, seed=1)
    d2 = make_synthetic(n=5, seed=2)
    assert not np.array_equal(d1["images"], d2["images"])


def test_dataset_length():
    data = make_synthetic(n=30, seed=0)
    ds = PathSegDataset(data)
    assert len(ds) == 30


def test_dataset_getitem_shape():
    data = make_synthetic(n=10, seed=0)
    ds = PathSegDataset(data)
    img, msk = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(msk, torch.Tensor)
    assert img.shape == (3, H, W)
    assert msk.shape == (1, H, W)


def test_dataset_getitem_value_range():
    data = make_synthetic(n=5, seed=0)
    ds = PathSegDataset(data)
    img, msk = ds[0]
    assert img.min() >= 0.0
    assert img.max() <= 1.0
    assert set(msk.unique().tolist()).issubset({0, 1})


def test_dataset_with_indices():
    data = make_synthetic(n=20, seed=0)
    ds = PathSegDataset(data, indices=[0, 2, 4])
    assert len(ds) == 3


def test_create_datasets_keys():
    data = make_synthetic(n=50, seed=42)
    datasets = create_datasets(data, test_size=0.2, val_size=0.15, seed=7)
    assert set(datasets.keys()) == {"train", "val", "test"}
    assert len(datasets["train"]) > 0
    assert len(datasets["val"]) > 0
    assert len(datasets["test"]) > 0


def test_create_datasets_non_overlapping_indices():
    data = make_synthetic(n=100, seed=42)
    datasets = create_datasets(data, test_size=0.2, val_size=0.15, seed=7)
    train_bytes = set(img.tobytes() for img in datasets["train"].images)
    test_bytes = set(img.tobytes() for img in datasets["test"].images)
    val_bytes = set(img.tobytes() for img in datasets["val"].images)
    assert len(train_bytes & test_bytes) == 0
    assert len(train_bytes & val_bytes) == 0
    assert len(test_bytes & val_bytes) == 0


def test_dataloaders_creation():
    data = make_synthetic(n=30, seed=42)
    datasets = create_datasets(data, test_size=0.2, val_size=0.15, seed=7)
    loaders = create_dataloaders(datasets, batch_size=4, val_batch_size=8)
    assert set(loaders.keys()) == {"train", "val", "test"}
    for batch in loaders["train"]:
        images, masks = batch
        assert images.shape[0] <= 4
        assert images.shape[1:] == (3, H, W)
        assert masks.shape[1:] == (1, H, W)
        break


def test_dataloader_shuffling():
    data = make_synthetic(n=30, seed=42)
    datasets = create_datasets(data, test_size=0.2, val_size=0.15, seed=7)
    n_train = len(datasets["train"])
    loaders = create_dataloaders(datasets, batch_size=n_train)
    train_loader = loaders["train"]
    first_epoch = next(iter(train_loader))[0].clone()
    second_epoch = next(iter(train_loader))[0].clone()
    assert not torch.equal(first_epoch, second_epoch)


def test_train_transform_output():
    image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (H, W), dtype=np.uint8)
    img_t, msk_t = train_transform(image, mask)
    assert img_t.shape == (3, H, W)
    assert msk_t.shape == (1, H, W)
    assert img_t.dtype == torch.float32
    assert msk_t.dtype == torch.float32


def test_val_transform_identity():
    image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (H, W), dtype=np.uint8)
    img_t1, msk_t1 = val_transform(image, mask)
    img_t2, msk_t2 = val_transform(image, mask)
    assert torch.equal(img_t1, img_t2)
    assert torch.equal(msk_t1, msk_t2)


def test_transform_twice_different_train():
    image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (H, W), dtype=np.uint8)
    img1, msk1 = train_transform(image, mask)
    img2, msk2 = train_transform(image, mask)
    different = not (torch.equal(img1, img2) and torch.equal(msk1, msk2))
    assert different


def test_get_transform():
    train_tf = get_transform("train")
    val_tf = get_transform("val")
    test_tf = get_transform("test")
    assert train_tf is train_transform
    assert val_tf is val_transform
    assert test_tf is val_transform


def test_prepare_data_keys():
    result = prepare_data(use_synthetic=True, n_synthetic=30, batch_size=4, val_batch_size=8)
    assert "loaders" in result
    assert "datasets" in result
    assert "n_classes" in result
    assert "metadata" in result
    assert result["n_classes"] == 1


def test_prepare_data_loader_batches():
    result = prepare_data(use_synthetic=True, n_synthetic=30, batch_size=4, val_batch_size=8)
    loaders = result["loaders"]
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    train_batches = list(loaders["train"])
    assert len(train_batches) > 0
    for images, masks in train_batches:
        assert images.shape[0] <= 4
        assert masks.shape[0] <= 4
        break


def test_prepare_data_metadata():
    result = prepare_data(use_synthetic=True, n_synthetic=50, seed=42)
    meta = result["metadata"]
    assert meta["n_samples"] == 50
    assert meta["tile_size"] == (H, W)
    assert "positive_fraction" in meta
    assert 0 < meta["positive_fraction"] < 1


def test_synthetic_variable_sizes():
    for n in [1, 5, 100]:
        data = make_synthetic(n=n, seed=0)
        assert len(data["images"]) == n
        assert len(data["masks"]) == n


def test_dataset_source_metadata():
    data = make_synthetic(n=5, seed=0)
    ds = PathSegDataset(data)
    assert "synthetic" in ds.source.lower()


def test_dataset_default_transform():
    data = make_synthetic(n=5, seed=0)
    ds = PathSegDataset(data)
    img, msk = ds[0]
    assert img.shape == (3, H, W)
    assert msk.shape == (1, H, W)


@pytest.mark.parametrize("test_size,val_size", [(0.1, 0.1), (0.3, 0.2), (0.5, 0.0)])
def test_create_datasets_various_splits(test_size, val_size):
    data = make_synthetic(n=50, seed=42)
    datasets = create_datasets(data, test_size=test_size, val_size=val_size, seed=7)
    n_train = len(datasets["train"])
    n_val = len(datasets["val"])
    n_test = len(datasets["test"])
    assert n_train + n_val + n_test == 50
    assert n_train >= 1


def test_train_transform_mask_binary():
    rng = np.random.default_rng(0)
    for _ in range(8):
        image = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        mask = rng.integers(0, 2, (H, W), dtype=np.uint8)
        _, msk_t = train_transform(image, mask)
        unique_vals = msk_t.unique().tolist()
        assert all(v in (0.0, 1.0) for v in unique_vals), (
            f"Mask has non-binary values after augmentation: {unique_vals}"
        )
