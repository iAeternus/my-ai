from dataclasses import asdict
from pathlib import Path
from typing import Any

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
import yaml


def from_yaml(path: str | Path) -> Config:
    """从 YAML 配置文件构建 Config"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return from_dict(data)


def from_dict(data: dict[str, Any]) -> Config:
    """从字典构建 Config，缺失字段使用 dataclass 默认值"""
    ds = data.get("dataset", {})
    m = data.get("model", {})
    opt = data.get("optimizer", {})
    t = data.get("train", {})
    es = t.get("early_stopping", {})
    r = data.get("runtime", {})
    e = data.get("experiment", {})

    task_raw = data.get("task", "link_prediction")
    loss_raw = t.get("loss_type", "margin_ranking")

    return Config(
        task=TaskType(task_raw),
        dataset=DatasetConfig(
            name=ds.get("name", "fb15k-237"),
            root=ds.get("root", "packages/kge/data"),
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
            save_dir=e.get("save_dir", "packages/kge/outputs"),
            seeds=e.get("seeds", [42]),
        ),
    )


def from_cli(
    yaml_path: str | Path | None,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """配置加载，优先级：代码默认值 < YAML < CLI

    --model 和 --dataset 可覆盖 YAML 中的模型和数据集。
    encoder 专属参数由 builder.py 提供默认值，无需在 YAML 中声明。
    """
    config = Config()
    if yaml_path:
        path = Path(yaml_path)
        if path.exists():
            config = from_yaml(yaml_path)

    if cli_overrides:
        raw = asdict(config)
        _apply_overrides(raw, cli_overrides)
        config = from_dict(raw)

    return config


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    """CLI 参数覆盖 YAML 配置（原地修改 raw dict），仅覆盖非 None 的项"""
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

    for cli_key, value in overrides.items():
        if value is None:
            continue
        path = _OVERRIDE_MAP.get(cli_key)
        if path is None:
            continue
        _set_nested(raw, path, value)


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """按 key 路径设置嵌套字典值，中间层不存在则自动创建"""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
