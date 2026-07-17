"""路径解析基础设施

通过 ``__file__`` 向上查找最顶层的 ``pyproject.toml`` 定位项目根目录，
提供不依赖 CWD 的绝对路径常量和路径解析工具
"""

from __future__ import annotations
from pathlib import Path


def _find_project_root() -> Path:
    """从当前文件向上查找最顶层的 ``pyproject.toml``，返回 monorepo 根目录。"""
    current = Path(__file__).resolve().parent
    root: Path | None = None
    for ancestor in [current, *current.parents]:
        if (ancestor / "pyproject.toml").is_file():
            root = ancestor
    if root is None:
        raise RuntimeError("找不到项目根目录（未找到 pyproject.toml）")
    return root


PROJECT_ROOT = _find_project_root()
"""项目根目录（绝对路径）—— monorepo 的 ``pyproject.toml`` 所在目录"""

DATA_DIR = PROJECT_ROOT / "data"
"""数据集根目录（绝对路径）"""

OUTPUT_DIR = PROJECT_ROOT / "outputs"
"""实验输出根目录（绝对路径）"""


def resolve_path(path: str | Path) -> Path:
    """将路径解析为绝对路径

    规则：
    - 绝对路径 -> 原样返回
    - 相对路径 -> 相对于 ``PROJECT_ROOT`` 解析
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()
