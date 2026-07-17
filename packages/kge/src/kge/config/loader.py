"""KGE 配置加载器。

委托给 ``core.config`` 的通用加载函数（``load_config_from_yaml``、
``load_config_from_cli``、``apply_overrides``、``set_nested``），
仅保留 KGE 特有的 ``from_dict`` 工厂函数和 ``_OVERRIDE_MAP`` 映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import load_config_from_cli, load_config_from_yaml

from kge.config.schema import (
    Config,
    DatasetConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    LossType,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskType,
    TrainConfig,
)
from kge.utils.paths import DATA_DIR, OUTPUT_DIR, PACKAGE_ROOT, resolve_path

# CLI arg name to nested dict path mapping (KGE-specific)

_OVERRIDE_MAP: dict[str, list[str]] = {
    "task": ["task"],
    "dataset": ["dataset", "name"],
    "root": ["dataset", "root"],
    "encoder": ["model", "encoder_name"],
    "head": ["model", "head_name"],
    "batch_size": ["dataset", "batch_size"],
    "negative_samples": ["dataset", "num_negative_samples"],
    "embedding_dim": ["model", "params", "embedding_dim"],
    "gamma": ["model", "params", "gamma"],
    "p_norm": ["model", "params", "p_norm"],
    "hidden_dim": ["model", "params", "hidden_dim"],
    "hidden_dropout": ["model", "params", "hidden_dropout"],
    "relation_dim": ["model", "params", "relation_dim"],
    "epsilon": ["model", "params", "epsilon"],
    "kernel_size": ["model", "params", "kernel_size"],
    "conv_out_channels": ["model", "params", "conv_out_channels"],
    "input_dropout": ["model", "params", "input_dropout"],
    "feature_dropout": ["model", "params", "feature_dropout"],
    "lr": ["optimizer", "params", "lr"],
    "weight_decay": ["optimizer", "params", "weight_decay"],
    "epochs": ["train", "epochs"],
    "loss_type": ["train", "loss_type"],
    "margin": ["train", "margin"],
    "label_smoothing": ["train", "label_smoothing"],
    "eval_interval": ["train", "eval_interval"],
    "patience": ["train", "early_stopping", "patience"],
    "device": ["runtime", "device"],
    "compile": ["runtime", "compile"],
    "seeds": ["experiment", "seeds"],
    "name_prefix": ["experiment", "name_prefix"],
    "save_dir": ["experiment", "save_dir"],
}


# Config loading (delegates to core)


def from_yaml(path: str | Path) -> Config:
    """从 YAML 配置文件构建 Config（委托给 core）。"""
    path = Path(path)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    return load_config_from_yaml(path, factory=from_dict)


def from_dict(data: dict[str, Any]) -> Config:
    """从字典构建 Config（KGE 特有工厂函数）。

    对路径字段做绝对路径解析。
    """
    ds: dict[str, Any] = dict(data.get("dataset", {}))
    m: dict[str, Any] = dict(data.get("model", {}))
    opt: dict[str, Any] = dict(data.get("optimizer", {}))
    t: dict[str, Any] = dict(data.get("train", {}))
    es: dict[str, Any] = dict(t.get("early_stopping", {}))
    r: dict[str, Any] = dict(data.get("runtime", {}))
    e: dict[str, Any] = dict(data.get("experiment", {}))

    task_raw: str = data.get("task", "link_prediction")
    loss_raw: str = t.get("loss_type", "margin_ranking")

    return Config(
        task=TaskType(task_raw),
        dataset=DatasetConfig(
            name=ds.get("name", "fb15k-237"),
            root=str(resolve_path(ds.get("root", str(DATA_DIR)))),
            batch_size=ds.get("batch_size", 1024),
            num_negative_samples=ds.get("num_negative_samples", 128),
            num_workers=ds.get("num_workers", 0),
        ),
        model=ModelConfig(
            encoder_name=m.get("encoder_name", "trans-e"),
            head_name=m.get("head_name", "link_prediction"),
            params=m.get("params", {}),
        ),
        optimizer=OptimizerConfig(
            name=opt.get("name", "adam"),
            params=opt.get("params", {}),
        ),
        train=TrainConfig(
            epochs=t.get("epochs", 500),
            lr=t.get("lr", 0.001),
            weight_decay=t.get("weight_decay", 0.0),
            loss_type=LossType(loss_raw),
            margin=t.get("margin", 1.0),
            adversarial_temperature=t.get("adversarial_temperature", 0.0),
            label_smoothing=t.get("label_smoothing", 0.0),
            regularization_weight=t.get("regularization_weight", 0.0),
            eval_interval=t.get("eval_interval", 10),
            eval_ks=t.get("eval_ks", [1, 3, 10]),
            early_stopping=EarlyStoppingConfig(
                enabled=es.get("enabled", True),
                patience=es.get("patience", 30),
                monitor=es.get("monitor", "val_mrr"),
                min_delta=es.get("min_delta", 0.0),
            ),
        ),
        runtime=RuntimeConfig(
            device=r.get("device", "auto"),
            compile=r.get("compile", "auto"),
        ),
        experiment=ExperimentConfig(
            name_prefix=e.get("name_prefix", "kge"),
            save_dir=str(resolve_path(e.get("save_dir", str(OUTPUT_DIR)))),
            seeds=e.get("seeds", [42]),
        ),
    )


def from_cli(
    yaml_path: str | Path | None,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """配置加载，优先级：代码默认值 < YAML < CLI（委托给 core）。

    --model 和 --dataset 可覆盖 YAML 中的模型和数据集。
    encoder 专属参数由 builder.py 提供默认值，无需在 YAML 中声明。
    """
    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.is_absolute():
            yaml_path = str(PACKAGE_ROOT / path)
    return load_config_from_cli(
        yaml_path,
        cli_overrides,
        factory=from_dict,
        override_map=_OVERRIDE_MAP,
        defaults=Config(),
    )


# 以下函数已删除，由 core.apply_overrides / core.set_nested 替代
