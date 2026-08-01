import pytest
import torch
from src.models.base import BaseBackbone
from src.utils.config import BEVConfig
from src.models.blocks import ConvBNReLU, ResidualBlock
from src.models.resnet import BEVResNet
from src.models.registry import ModelRegistry
import src.models.resnet  # triggers registration
from src.data.dataset import BEVDataset

def test_config_defaults():
    cfg = BEVConfig()
    assert cfg.input_size == (256, 256)
    assert cfg.num_stages == len(cfg.strides)
    assert cfg.batch_size == 8


def test_config_custom():
    cfg = BEVConfig(batch_size=16, lr=1e-4)
    assert cfg.batch_size == 16
    assert cfg.lr == 1e-4


def test_config_invalid_strides():
    with pytest.raises(ValueError, match="strides"):
        BEVConfig(num_stages=4, strides=(1, 2))


def test_config_invalid_batch():
    with pytest.raises(ValueError, match="batch_size"):
        BEVConfig(batch_size=-1)


def test_config_invalid_device():
    with pytest.raises(ValueError, match="device"):
        BEVConfig(device="gpu")

def test_cannot_instantiate_base():
    """ABC cannot be instantiated directly."""
    cfg = BEVConfig()
    with pytest.raises(TypeError):
        BaseBackbone(cfg)


def test_subclass_must_implement_forward():
    """Subclass missing forward() cannot be instantiated."""
    class IncompleteBackbone(BaseBackbone):
        def get_output_channels(self):
            return {"C4": 256}
        # forward() not implemented

    cfg = BEVConfig()
    with pytest.raises(TypeError):
        IncompleteBackbone(cfg)

def test_convbnrelu_shape():
    block = ConvBNReLU(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 256, 256)   # batch=2, C=3, H=256, W=256
    out = block(x)
    assert out.shape == (2, 64, 256, 256)

def test_residual_block_same_shape():
    block = ResidualBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 64, 64)
    out = block(x)
    assert out.shape == (2, 64, 64, 64)

def test_residual_block_downsample():
    block = ResidualBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 64, 64)
    out = block(x)
    assert out.shape == (2, 128, 32, 32)  # stride=2 halves H and W
def test_bevresnet_output_keys():
    cfg = BEVConfig()
    model = BEVResNet(cfg)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert set(out.keys()) == {"C2", "C3", "C4", "C5"}

def test_bevresnet_output_shapes():
    cfg = BEVConfig()
    model = BEVResNet(cfg)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out["C2"].shape == (2, 64,  128, 128)  # stem halves: 256→128, stage1 stride=1
    assert out["C3"].shape == (2, 128, 64,  64)   # stage2 stride=2
    assert out["C4"].shape == (2, 256, 32,  32)   # stage3 stride=2
    assert out["C5"].shape == (2, 512, 16,  16)   # stage4 stride=2

def test_bevresnet_param_count():
    cfg = BEVConfig()
    model = BEVResNet(cfg)
    assert model.num_parameters() > 0
    print(f"\nTotal params: {model.num_parameters():,}")

def test_bevresnet_fulfills_contract():
    """BEVResNet must satisfy BaseBackbone interface."""
    cfg = BEVConfig()
    model = BEVResNet(cfg)
    assert isinstance(model, BaseBackbone)
    ch = model.get_output_channels()
    assert ch["C5"] == 512

def test_registry_contains_bevresnet():
    assert "bevresnet" in ModelRegistry.available()

def test_registry_get_and_instantiate():
    cfg = BEVConfig()
    cls = ModelRegistry.get("bevresnet")
    model = cls(cfg)
    assert isinstance(model, BaseBackbone)

def test_registry_missing_key():
    with pytest.raises(KeyError):
        ModelRegistry.get("nonexistent")
def test_dataset_len():
    ds = BEVDataset(root="data/", split="train")
    assert len(ds) == 50000

def test_dataset_getitem_shape():
    ds = BEVDataset(root="data/", split="train")
    img, label = ds[0]
    assert img.shape == (3, 256, 256)
    assert isinstance(label, int)

def test_dataset_val_split():
    ds = BEVDataset(root="data/", split="val")
    assert len(ds) == 10000

def test_dataset_invalid_split():
    with pytest.raises(ValueError, match="split"):
        BEVDataset(root="data/", split="test")

def test_dataset_label_map():
    ds = BEVDataset(root="data/", split="train")
    assert ds.class_name(0) == "airplane"
    assert ds.class_name(9) == "truck"

def test_dataset_repr():
    ds = BEVDataset(root="data/", split="train")
    assert "BEVDataset" in repr(ds)

from src.data.dataset import build_dataloader

def test_dataloader_batch_shape():
    dl = build_dataloader(split="train", batch_size=4)
    images, labels = next(iter(dl))
    assert images.shape == (4, 3, 256, 256)
    assert labels.shape == (4,)

def test_dataloader_val():
    dl = build_dataloader(split="val", batch_size=4)
    images, labels = next(iter(dl))
    assert images.shape == (4, 3, 256, 256)