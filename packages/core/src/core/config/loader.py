from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from core.config.base import SerializableConfig
import yaml

T = TypeVar("T", bound=SerializableConfig)


def load_config_from_yaml(
    path: str | Path,
    *,
    factory: Callable[[dict[str, Any]], T],
) -> T:
    """读取 YAML 配置文件，通过 ``factory`` 构建目标 Config 对象

    Args:
        path: YAML 文件路径
        factory: ``dict -> Config`` 的构建函数（通常为各包的 ``from_dict``）

    Returns:
        由 ``factory`` 返回的 Config 对象
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return factory(data)


def load_config_from_dict(
    data: dict[str, Any],
    *,
    factory: Callable[[dict[str, Any]], T],
) -> T:
    """从字典构建 Config 对象

    直接委托给 ``factory``，作为统一入口存在，方便后续扩展校验/日志

    Args:
        data: 配置字典
        factory: ``dict -> Config`` 的构建函数

    Returns:
        由 ``factory`` 返回的 Config 对象
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
    """分层加载配置：**默认值 -> YAML -> CLI 覆盖**

    Args:
        yaml_path: YAML 配置文件路径（可为 ``None``）
        overrides: CLI 参数覆盖字典（``{arg_name: value, ...}``）
        factory: ``dict -> Config`` 的构建函数
        override_map: CLI 参数名到嵌套字典路径的映射（如 ``{"hidden_dim": ["model", "params", "hidden_dim"]}``）
        defaults: 带全部默认值的 Config 实例（如 ``Config()``）

    Returns:
        合并后的 Config 对象
    """
    # 从默认配置开始
    config = defaults if defaults is not None else factory({})

    # 叠加 YAML
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            config = load_config_from_yaml(yaml_path, factory=factory)

    # 应用 CLI 覆盖
    if overrides:
        raw = asdict(config)
        apply_overrides(raw, overrides, override_map)
        config = factory(raw)

    return config


def set_nested(d: dict, keys: list[str], value: Any) -> None:
    """按 key 路径设置嵌套字典值，中间层不存在则自动创建

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
    """将 CLI 参数按 ``override_map`` 注入嵌套字典（原地修改）

    仅处理非 ``None`` 且在 ``override_map`` 中存在的键

    Args:
        data: 目标配置字典（原地修改）
        overrides: CLI 覆盖参数
        override_map: CLI 参数名到嵌套字典路径的映射
    """
    for cli_key, value in overrides.items():
        if value is None:
            continue
        path = override_map.get(cli_key)
        if path is None:
            continue
        set_nested(data, path, value)
