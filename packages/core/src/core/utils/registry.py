from typing import Any, Callable, Generic, Iterator, TypeVar
import torch.nn as nn

T = TypeVar("T")


class Registry(Generic[T]):
    """统一注册器

    Example:
        >>> MODELS = Registry[type[nn.Module]](
        ...     "model",
        ...     base_class=nn.Module,
        ... )

        >>> @MODELS.register()
        ... class GCN(nn.Module):
        ...     ...

        >>> model = MODELS.build("gcn", hidden_dim=64)
    """

    def __init__(
        self,
        kind: str = "item",
        *,
        base_class: type | None = None,
    ) -> None:
        self._kind = kind
        self._base_class = base_class
        self._registry: dict[str, T] = {}
        self._frozen = False

    def register(
        self,
        name: str | None = None,
        *,
        replace: bool = False,
    ) -> Callable[[T], T]:
        """注册对象"""

        def decorator(obj: T) -> T:
            if self._frozen:
                raise RuntimeError(f"{self._kind} registry has been frozen")

            key = self._normalize_key(name or getattr(obj, "__name__"))
            if key in self._registry and not replace:
                raise KeyError(f"{self._kind!r} '{key}' has already been registered.")
            if (
                self._base_class is not None
                and isinstance(obj, type)
                and not issubclass(obj, self._base_class)
            ):
                raise TypeError(
                    f"{obj.__name__} must inherit from " f"{self._base_class.__name__}."
                )

            self._registry[key] = obj
            return obj

        return decorator

    # MMEngine 风格兼容
    register_module = register

    def get(self, name: str) -> T:
        """获取注册对象（查询键自动标准化为小写 + 连字符）。"""
        key = self._normalize_key(name)
        try:
            return self._registry[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._registry))
            raise KeyError(
                f"Unknown {self._kind}: '{name}'. "
                f"Available {self._kind}s: [{available}]"
            ) from exc

    @staticmethod
    def _normalize_key(name: str) -> str:
        return name.replace("_", "-").lower()

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """实例化注册对象"""
        builder = self.get(name)
        return builder(*args, **kwargs)

    def unregister(self, name: str) -> T:
        """注销对象"""
        if self._frozen:
            raise RuntimeError(f"{self._kind} registry has been frozen.")

        try:
            return self._registry.pop(name)
        except KeyError as exc:
            raise KeyError(f"Unknown {self._kind}: '{name}'.") from exc

    def clear(self) -> None:
        """清空注册表"""
        if self._frozen:
            raise RuntimeError(f"{self._kind} registry has been frozen.")
        self._registry.clear()

    def freeze(self) -> None:
        """冻结注册表"""
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def __getitem__(self, name: str) -> T:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()

    @property
    def available(self) -> list[str]:
        """所有已注册名称"""
        return sorted(self._registry)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"kind={self._kind!r}, "
            f"size={len(self)}, "
            f"frozen={self._frozen}, "
            f"items={self.available})"
        )


MODEL_REGISTRY = Registry[type[nn.Module]]("model", base_class=nn.Module)
# DATASET_REGISTRY = Registry[type[BaseDataset]]("dataset", base_class=BaseDataset)
LOSS_REGISTRY = Registry[type[nn.Module]]("loss", base_class=nn.Module)
METRIC_REGISTRY = Registry[Callable]("metric")
TRANSFORM_REGISTRY = Registry[type]("transform")
