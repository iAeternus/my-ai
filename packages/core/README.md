# core

my-ai 公共基础设施。GNN、KGE 等子包共享的抽象与工具，不包含任何领域逻辑。

## 配置

继承 base dataclass，实现 `from_dict` 工厂，组合成 `Config`

```python
from dataclasses import dataclass, field
from core.config import (
    BaseRuntimeConfig, BaseExperimentConfig, BaseEarlyStoppingConfig,
    BaseOptimizerConfig, SerializableConfig, validate_monitor,
    load_config_from_yaml, load_config_from_cli,
)
from core.utils import MONITOR_MODES

@dataclass(slots=True, frozen=True)
class RuntimeConfig(BaseRuntimeConfig):
    pass  # 继承 device="auto", compile="auto"

@dataclass(slots=True, frozen=True)
class ExperimentConfig(BaseExperimentConfig):
    pass  # 字段由 from_dict 设置包特有默认值

@dataclass(slots=True, frozen=True)
class MyConfig(SerializableConfig):
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        validate_monitor(self.train.monitor, monitor_modes=MONITOR_MODES)

# 加载：默认值 < YAML < CLI 覆盖
cfg = load_config_from_cli("config/baseline.yaml", {"lr": 0.01},
    factory=from_dict, override_map={"lr": ["train", "lr"]}, defaults=MyConfig())
```

## 数据

继承 `BaseDataset`，实现 `_load()`，用 `@ensure_loaded` 守卫延迟属性

```python
from core.datasets import BaseDataset, ensure_loaded, DATASET_REGISTRY

@DATASET_REGISTRY.register("my-data")
class MyDataset(BaseDataset):
    def _load(self):
        # 下载、解析、预处理 → 填充 self._data
        self._data = ...

    @property
    @ensure_loaded
    def num_train(self) -> int: return len(self._data["train"])

    def __len__(self): return self.num_train
    def __getitem__(self, idx): return self._data["train"][idx]
```

辅助工具：`download_file`、`extract_tar`、`extract_zip`、`random_split_indices`、`stratified_split_indices`

## 模型 / 损失 / 指标

`Registry` 泛型容器，支持装饰器注册和按名构建

```python
from core import Registry
import torch.nn as nn

ENCODER_REGISTRY = Registry[type[nn.Module]]("encoder", base_class=nn.Module)

@ENCODER_REGISTRY.register("mlp")
class MLPEncoder(nn.Module): ...

encoder = ENCODER_REGISTRY.build("mlp", hidden_dim=256)
```

core 预置 `MODEL_REGISTRY`、`LOSS_REGISTRY`、`METRIC_REGISTRY`、`TRANSFORM_REGISTRY`

## 训练

`BaseTrainer` 为参考实现，子包可自定训练循环。必须复用的独立组件：

```python
from core.utils import get_device, seed_everything, setup_logging
from core.utils import EarlyStopping, CheckpointManager, MONITOR_MODES
from core.trainer import should_compile, OPTIMIZER_REGISTRY

device = get_device("auto")
seed_everything(42)
setup_logging("outputs/run/logs")

optimizer = OPTIMIZER_REGISTRY["adam"](model.parameters(), lr=0.001)

if should_compile(cfg.runtime.compile, device):
    model = torch.compile(model)

early_stop = EarlyStopping(patience=30, mode=MONITOR_MODES["val_loss"])
ckpt = CheckpointManager("outputs/run/checkpoints")

for epoch in range(epochs):
    loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    if early_stop.step(val_loss): break
    if early_stop.improved:
        ckpt.save(model, optimizer, epoch, val_loss)

ckpt.load(model)
```

## 实验管理

`ExperimentManager` 统一管理产物目录、配置保存、历史记录、绘图和汇总

```python
from core.experiment import ExperimentManager, PlotSpec

exp = ExperimentManager(
    save_dir="outputs", name_prefix="exp",
    dir_segments=["fb15k", "rotate"], seeds=[42, 123],
)
exp.setup_multi() if exp.is_multi else exp.setup()
exp.save_config(cfg.to_dict())
setup_logging(exp.log_dir)

for seed in cfg.experiment.seeds:
    run_dir = exp.seed_run_dir(seed) if exp.is_multi else exp.root_dir
    history = train(...)
    exp.save_history(history, run_dir=run_dir)
    exp.plot_metrics(history, specs=[
        PlotSpec(title="Loss", train_key="loss", val_key="val_loss", ylabel="Loss"),
        PlotSpec(title="MRR", multi_keys=["val_mrr", "test_mrr"], ylabel="MRR"),
    ], run_dir=run_dir)
```

## 路径

`get_package_root` + `PackagePaths` 统一管理包内路径

```python
from core.utils.paths import get_package_root, PackagePaths, PROJECT_ROOT

_pkg = PackagePaths(get_package_root(__file__))
PACKAGE_ROOT = _pkg.root       # packages/my-pkg/
DATA_DIR    = _pkg.data_dir    # packages/my-pkg/data/
OUTPUT_DIR  = _pkg.output_dir  # packages/my-pkg/outputs/

config_path = _pkg.resolve("config/baseline.yaml")
```

- `PROJECT_ROOT`：monorepo 根（最顶层 `pyproject.toml`）
- `get_package_root(__file__)`：调用方最近的 `pyproject.toml`

## 回调

core 提供 `Callback` 基类和两个常用回调

```python
from core.trainer import Callback, EarlyStoppingCallback, CheckpointCallback

trainer = BaseTrainer(model, optimizer, callbacks=[
    EarlyStoppingCallback(EarlyStopping(patience=30)),
    CheckpointCallback(CheckpointManager("ckpt")),
])
trainer.train(train_loader, epochs=100)
```
