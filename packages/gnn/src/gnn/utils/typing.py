from typing import TypeVar, cast

T = TypeVar("T")


def get_param(params: dict[str, object], key: str, default: T) -> T:
    value = params.get(key, default)
    return cast(T, value)


def pop_config(params: dict[str, object], key: str, default: T) -> T:
    return cast(T, params.pop(key, default))
