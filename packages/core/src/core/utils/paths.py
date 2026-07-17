"""路径解析基础设施

通过 ``__file__`` 向上查找最顶层的 ``pyproject.toml`` 定位 monorepo 根目录，
同时提供按**最近** ``pyproject.toml`` 定位子包根目录的能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ── 项目（monorepo）根 ──────────────────────────────────────────────────


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
"""monorepo 根目录（绝对路径）——最顶层 ``pyproject.toml`` 所在目录"""

# [deprecated] 以下两个模块级常量仅指向 monorepo 根，不推荐直接使用。
# 各子包应使用 ``PackagePaths`` 获取包级 data / outputs 目录。
DATA_DIR = PROJECT_ROOT / "data"
""".. deprecated:: 使用 ``PackagePaths.data_dir`` 替代"""

OUTPUT_DIR = PROJECT_ROOT / "outputs"
""".. deprecated:: 使用 ``PackagePaths.output_dir`` 替代"""


# ── 子包根 ─────────────────────────────────────────────────────────────


def get_package_root(caller_file: str | Path) -> Path:
    """从调用方文件向上查找**最近**的 ``pyproject.toml``，返回子包根目录。

    与 ``_find_project_root()``（找最顶层 → monorepo 根）不同，
    此函数返回离 *caller_file* 最近的那个。

    Args:
        caller_file: 调用方的 ``__file__``（如 ``paths.py`` 中传入 ``__file__``）。

    Returns:
        子包根目录（绝对路径）。

    Example:
        >>> # 在 packages/kge/src/kge/utils/paths.py 中调用：
        >>> get_package_root(__file__)
        Path('.../packages/kge')
    """
    current = Path(caller_file).resolve().parent
    for ancestor in [current, *current.parents]:
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise RuntimeError(
        f"找不到包根目录（从 {caller_file} 向上未找到 pyproject.toml）"
    )


@dataclass
class PackagePaths:
    """子包路径配置。

    每个子包（GNN、KGE 等）用 ``get_package_root(__file__)`` 初始化一个实例，
    统一管理包内的 data / outputs / config 目录。

    Example:
        >>> _pkg = PackagePaths(get_package_root(__file__))
        >>> _pkg.data_dir   # <package>/data/
        >>> _pkg.resolve("config/baseline.yaml")  # <package>/config/baseline.yaml
    """

    root: Path
    """包根目录（包含 ``pyproject.toml``）"""

    @property
    def data_dir(self) -> Path:
        """数据集目录：``<root>/data/``"""
        return self.root / "data"

    @property
    def output_dir(self) -> Path:
        """实验输出目录：``<root>/outputs/``"""
        return self.root / "outputs"

    @property
    def config_dir(self) -> Path:
        """配置文件目录：``<root>/config/``"""
        return self.root / "config"

    def resolve(self, path: str | Path) -> Path:
        """解析路径。

        规则：
        - 绝对路径 → 原样返回
        - 相对路径 → 相对于 ``self.root`` 解析为绝对路径
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.root / p).resolve()


# ── 通用路径解析（保留向后兼容）─────────────────────────────────────────


def resolve_path(path: str | Path) -> Path:
    """将路径解析为绝对路径（相对于 ``PROJECT_ROOT``）。

    .. deprecated:: 子包内请用 ``PackagePaths.resolve()`` 或包内的 ``resolve_path()``。

    规则：
    - 绝对路径 -> 原样返回
    - 相对路径 -> 相对于 ``PROJECT_ROOT`` 解析
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()
