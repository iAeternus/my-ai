"""所有数据集的根抽象

提供 lazy-loading 装饰器和跨领域共享的最小接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable


def ensure_loaded(func: Callable) -> Callable:
    """Lazy-load guard 装饰器：首次访问自动触发 ``_load()``

    来源：kge.datasets.base，泛化后迁移至 core
    """

    @wraps(func)
    def wrapper(self: BaseDataset, *args, **kwargs):
        if not self._loaded:
            self._load()
            self._loaded = True
        return func(self, *args, **kwargs)

    return wrapper


class BaseDataset(Dataset, ABC):
    """所有数据集的根抽象类

    子类需实现 ``_load()``，在其中完成数据加载与预处理，
    其他属性通过 ``@ensure_loaded`` 保证懒加载安全

    Attributes:
        root: 数据集根目录
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self._loaded = False

    @abstractmethod
    def _load(self) -> None:
        """子类实现：加载数据到内部存储"""
        ...

    @property
    @ensure_loaded
    def num_train(self) -> int:
        """训练集样本数，子类可覆盖此属性以提供具体实现"""
        raise NotImplementedError

    @property
    @ensure_loaded
    def num_val(self) -> int:
        """验证集样本数"""
        raise NotImplementedError

    @property
    @ensure_loaded
    def num_test(self) -> int:
        """测试集样本数"""
        raise NotImplementedError
