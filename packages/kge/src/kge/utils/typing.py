"""类型与参数提取工具"""

from typing import TypeVar, cast

T = TypeVar("T")


def dict_get_or_default(params: dict[str, object], key: str, default: T) -> T:
    value = params.get(key, default)
    return cast(T, value)


def dict_pop_or_default(params: dict[str, object], key: str, default: T) -> T:
    return cast(T, params.pop(key, default))
