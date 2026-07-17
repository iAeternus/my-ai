"""配置基础设施。

提供可复用的 base dataclass 和纯函数形式的配置加载工具，
通过 ``factory`` 回调适配任意包的具体 Config 类。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

# ── 可复用的 base dataclass ──────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class BaseRuntimeConfig:
    """运行时配置基类（device + compile）。"""

    device: str = "auto"
    compile: str | bool = "auto"


@dataclass(slots=True, frozen=True)
class BaseExperimentConfig:
    """实验配置基类（名称、保存目录、随机种子）。"""

    name_prefix: str = "experiment"
    save_dir: str = ""
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass(slots=True, frozen=True)
class BaseEarlyStoppingConfig:
    """早停配置基类。"""

    enabled: bool = True
    patience: int = 30
    monitor: str = "val_loss"
    min_delta: float = 0.0


# ── 纯函数：配置加载 ─────────────────────────────────────────────────


def load_config_from_yaml(
    path: str | Path,
    *,
    factory: Callable[[dict[str, Any]], T],
) -> T:
    """读取 YAML 配置文件，通过 ``factory`` 构建目标 Config 对象。

    Args:
        path: YAML 文件路径。
        factory: ``dict -> Config`` 的构建函数（通常为各包的 ``from_dict``）。

    Returns:
        由 ``factory`` 返回的 Config 对象。
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return factory(data)


def load_config_from_dict(
    data: dict[str, Any],
    *,
    factory: Callable[[dict[str, Any]], T],
) -> T:
    """从字典构建 Config 对象。

    直接委托给 ``factory``，作为统一入口存在，方便后续扩展校验/日志。

    Args:
        data: 配置字典。
        factory: ``dict -> Config`` 的构建函数。

    Returns:
        由 ``factory`` 返回的 Config 对象。
    """
    return factory(data)


def load_config_from_cli(
    yaml_path: str | Path | None,
    overrides: dict[str, Any] | None,
    *,
    factory: Callable[[dict[str, Any]], T],
    override_map: dict[str, list[str]],
    defaults: T | None = None,
) -> T:
    """分层加载配置：**默认值 -> YAML -> CLI 覆盖**。

    Args:
        yaml_path: YAML 配置文件路径（可为 ``None``）。
        overrides: CLI 参数覆盖字典（``{arg_name: value, ...}``）。
        factory: ``dict -> Config`` 的构建函数。
        override_map: CLI 参数名到嵌套字典路径的映射
            （如 ``{"hidden_dim": ["model", "params", "hidden_dim"]}``）。
        defaults: 带全部默认值的 Config 实例（如 ``Config()``）。

    Returns:
        合并后的 Config 对象。
    """
    # 1. 从默认配置开始
    config = defaults if defaults is not None else factory({})

    # 2. 叠加 YAML
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            config = load_config_from_yaml(yaml_path, factory=factory)

    # 3. 应用 CLI 覆盖
    if overrides:
        raw = asdict(config)
        apply_overrides(raw, overrides, override_map)
        config = factory(raw)

    return config


# ── 纯函数：字典嵌套操作 ─────────────────────────────────────────────


def set_nested(d: dict, keys: list[str], value: Any) -> None:
    """按 key 路径设置嵌套字典值，中间层不存在则自动创建。

    Example:
        >>> d = {}
        >>> set_nested(d, ["model", "params", "hidden_dim"], 128)
        >>> d
        {'model': {'params': {'hidden_dim': 128}}}
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def apply_overrides(
    data: dict[str, Any],
    overrides: dict[str, Any],
    override_map: dict[str, list[str]],
) -> None:
    """将 CLI 参数按 ``override_map`` 注入嵌套字典（原地修改）。

    仅处理非 ``None`` 且在 ``override_map`` 中存在的键。

    Args:
        data: 目标配置字典（原地修改）。
        overrides: CLI 覆盖参数。
        override_map: CLI 参数名到嵌套字典路径的映射。
    """
    for cli_key, value in overrides.items():
        if value is None:
            continue
        path = override_map.get(cli_key)
        if path is None:
            continue
        set_nested(data, path, value)
