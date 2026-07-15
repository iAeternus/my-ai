import importlib
import pkgutil

from kge.datasets.base import BaseKGDataset
from kge.datasets.registry import KGDatasetRegistry

_MODULE_BLACKLIST = {"base", "registry"}
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name in _MODULE_BLACKLIST or _module_info.name.startswith("_"):
        continue
    importlib.import_module(f"kge.datasets.{_module_info.name}")


def load_dataset(name: str, root: str = "packages/kge/data") -> BaseKGDataset:
    dataset_cls = KGDatasetRegistry.get(name)
    return dataset_cls(root=root)


__all__ = [
    "BaseKGDataset",
    "KGDatasetRegistry",
    "load_dataset",
]
