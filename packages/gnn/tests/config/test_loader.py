from pathlib import Path

import pytest

from gnn.config.loader import from_dict, from_yaml
from gnn.config.schema import Config

YAML_CONTENT = """
dataset:
  name: cora
  root: packages/gnn/data

model:
  name: gcn
  params:
    hidden_dim: 16
    dropout: 0.5
    num_layers: 2

optimizer:
  name: adam
  params:
    lr: 0.01
    weight_decay: 0.0005

scheduler:
  enabled: false

train:
  epochs: 100
  early_stopping:
    enabled: true
    patience: 30
    monitor: val_loss

runtime:
  device: auto
  compile: auto

experiment:
  name_prefix: gcn-cora-baseline
  save_dir: packages/gnn/outputs
  seeds:
    - 42
"""


@pytest.fixture
def gcn_yaml(tmp_path: Path) -> Path:
    """创建临时实验配置"""
    path = tmp_path / "gcn-cora.yaml"
    path.write_text(YAML_CONTENT, encoding="utf-8")
    return path


def test_from_yaml(gcn_yaml: Path):
    """YAML能够正确解析"""

    config = from_yaml(gcn_yaml)
    assert isinstance(config, Config)

    # dataset
    assert config.dataset.name == "cora"
    assert config.dataset.root == "packages/gnn/data"

    # model
    assert config.model.name == "gcn"
    assert config.model.params == {
        "hidden_dim": 16,
        "dropout": 0.5,
        "num_layers": 2,
    }

    # optimizer
    assert config.optimizer.name == "adam"
    assert config.optimizer.params["lr"] == 0.01
    assert config.optimizer.params["weight_decay"] == 0.0005

    # train
    assert config.train.epochs == 100
    assert config.train.early_stopping.patience == 30
    assert config.train.early_stopping.monitor == "val_loss"

    # runtime
    assert config.runtime.device == "auto"
    assert config.runtime.compile == "auto"

    # experiment
    assert config.experiment.seeds == [42]


def test_from_dict_default_values():
    """缺省字段应该使用dataclass默认值"""
    config = from_dict({"dataset": {"name": "cora"}, "model": {"name": "gcn"}})

    assert config.dataset.name == "cora"
    assert config.dataset.root == "packages/gnn/data"
    assert config.model.params == {}
    assert config.optimizer.name == "adam"
    assert config.train.epochs == 100
    assert config.runtime.device == "auto"
