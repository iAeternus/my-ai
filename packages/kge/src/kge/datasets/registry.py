from __future__ import annotations


class KGDatasetRegistry:
    """KG 数据集注册中心

    用法：
        @KGDatasetRegistry.register("fb15k-237")
        class FB15k237Dataset(BaseKGDataset): ...
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册数据集类"""

        def decorator(dataset_cls: type):
            dataset_cls.name = name
            cls._registry[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(f"未知数据集: {name!r}。可用: {cls.available()}")
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry.keys())
