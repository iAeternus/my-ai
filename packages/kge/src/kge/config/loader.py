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
    """从yaml配置文件构建配置"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return from_dict(data)


def from_dict(data: dict[str, Any]) -> Config:
    """从字典构建配置"""
    # dataset
    ds = data.get("dataset", {})
    dataset_config = DatasetConfig(
        name=ds.get("name", "fb15k-237"),
        root=ds.get("root", "packages/kge/data"),
        batch_size=ds.get("batch_size", 1024),
        num_negative_samples=ds.get("num_negative_samples", 128),
        num_workers=ds.get("num_workers", 0),
    )

    # model
    m = data.get("model", {})
    model_config = ModelConfig(
        encoder_name=m.get("encoder_name", "trans-e"),
        head_name=m.get("head_name", "link_prediction"),
        params=m.get("params", {}),
    )

    # optimizer
    opt = data.get("optimizer", {})
    optimizer_config = OptimizerConfig(
        name=opt.get("name", "adam"),
        params=opt.get("params", {}),
    )

    # train
    t = data.get("train", {})
    es = t.get("early_stopping", {})
    loss_raw = t.get("loss_type", "margin_ranking")
    train_config = TrainConfig(
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
    )

    # task
    task_raw = data.get("task", "link_prediction")
    task_config = TaskType(task_raw)

    # runtime
    r = data.get("runtime", {})
    runtime_config = RuntimeConfig(
        device=r.get("device", "auto"),
        compile=r.get("compile", "auto"),
    )

    # experiment
    e = data.get("experiment", {})
    experiment_config = ExperimentConfig(
        name_prefix=e.get("name_prefix", "kge"),
        save_dir=e.get("save_dir", "packages/kge/outputs"),
        seeds=e.get("seeds", [42]),
    )

    return Config(
        task=task_config,
        dataset=dataset_config,
        model=model_config,
        optimizer=optimizer_config,
        train=train_config,
        runtime=runtime_config,
        experiment=experiment_config,
    )


def from_cli(
    yaml_path: str | Path,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """配置优先级：代码默认值 < defaults/{encoder}.yaml < 指定 YAML < CLI"""
    path = Path(yaml_path)
    if path.exists():
        cfg = from_yaml(yaml_path)
    else:
        cfg = from_dict({})

    defaults_data = model_defaults(cfg.model.encoder_name, cfg.task)
    if defaults_data:
        defaults_cfg = from_dict(defaults_data)
        cfg = _merge_configs(defaults_cfg, cfg)  # cfg优先

    if cli_overrides:
        cfg = _apply_overrides(cfg, cli_overrides)
    return cfg


def model_defaults(
    encoder_name: str, task: TaskType, config_dir: str = "config"
) -> dict[str, Any]:
    """加载 encoder 的默认超参文件 config/{task}/defaults/{encoder}.yaml"""
    defaults_path = Path(config_dir) / task.value / "defaults" / f"{encoder_name}.yaml"
    if not defaults_path.exists():
        return {}
    with defaults_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_configs(base: Config, override: Config) -> Config:
    """用 override 覆盖 base 中的非默认值字段"""
    base_dict = base.to_dict()
    ov_dict = override.to_dict()
    merged = _deep_merge(base_dict, ov_dict)
    return from_dict(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 中的值优先"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """将 CLI 参数字典合并到 Config"""
    cfg_dict = cfg.to_dict()
    merged = _deep_merge(cfg_dict, _cli_to_nested(overrides))
    return from_dict(merged)


def _cli_to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """将扁平的 CLI 参数字典转为嵌套结构

    CLI 参数格式：
        --encoder trans-e   -> {"model": {"encoder_name": "trans-e"}}
        --embedding-dim 200 -> {"model": {"params": {"embedding_dim": 200}}}
        --lr 0.01           -> {"train": {"lr": 0.01}}
    """
    result: dict[str, Any] = {}

    # 直接映射
    key_mapping = {
        "encoder": ("model", "encoder_name"),
        "head": ("model", "head_name"),
        "dataset": ("dataset", "name"),
        "task": ("task", None),  # top-level
        "epochs": ("train", "epochs"),
        "lr": ("train", "lr"),
        "weight_decay": ("train", "weight_decay"),
        "loss_type": ("train", "loss_type"),
        "margin": ("train", "margin"),
        "batch_size": ("dataset", "batch_size"),
        "negative_samples": ("dataset", "num_negative_samples"),
        "seeds": ("experiment", "seeds"),
        "save_dir": ("experiment", "save_dir"),
        "name_prefix": ("experiment", "name_prefix"),
    }

    # encoder params 映射
    param_keys = {
        "embedding_dim",
        "gamma",
        "p_norm",
        "epsilon",
        "kernel_size",
        "conv_out_channels",
        "hidden_dropout",
        "input_dropout",
        "feature_dropout",
        "hidden_dim",
        "relation_dim",
        "pretrained_encoder",
    }

    for k, v in flat.items():
        if v is None:
            continue
        k_norm = k.replace("-", "_")
        if k_norm in key_mapping:
            section, field = key_mapping[k_norm]
            if section is None:
                result[field] = v
            else:
                result.setdefault(section, {})[field] = v
        elif k_norm in param_keys:
            result.setdefault("model", {}).setdefault("params", {})[k_norm] = v
        elif k_norm == "compile":
            result.setdefault("runtime", {})["compile"] = v
        elif k_norm == "device":
            result.setdefault("runtime", {})["device"] = v
        elif k_norm == "eval_interval":
            result.setdefault("train", {})["eval_interval"] = v

    return result
