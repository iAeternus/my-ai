import importlib
import pkgutil

from kge.utils.paths import DATA_DIR
from kge.datasets.base import BaseKGDataset
from kge.datasets.registry import KG_DATASET_REGISTRY
from kge.datasets.data_module import KGDataModule, KGBatch
from kge.datasets.sampler import (
    NegativeSampler,
    UniformNegativeSampler,
    BernoulliNegativeSampler,
    SelfAdversarialNegativeSampler,
)

_MODULE_BLACKLIST = {"base", "registry", "data_module", "sampler"}
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name in _MODULE_BLACKLIST or _module_info.name.startswith("_"):
        continue
    importlib.import_module(f"kge.datasets.{_module_info.name}")


def load_dataset(name: str, root: str = str(DATA_DIR)) -> BaseKGDataset:
    dataset_cls = KG_DATASET_REGISTRY[name]
    return dataset_cls(root=root)


__all__ = [
    "BaseKGDataset",
    "KG_DATASET_REGISTRY",
    "KGDataModule",
    "KGBatch",
    "NegativeSampler",
    "UniformNegativeSampler",
    "BernoulliNegativeSampler",
    "SelfAdversarialNegativeSampler",
    "load_dataset",
]
