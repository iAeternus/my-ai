"""GNN 包路径解析（委托给 core.utils.paths）。

通过 core 的 ``get_package_root()`` 定位 GNN 包根目录，
提供包级路径常量和配置/数据路径解析函数。
"""

from __future__ import annotations

from pathlib import Path

from core.utils.paths import PROJECT_ROOT, PackagePaths, get_package_root

# GNN package paths

_pkg = PackagePaths(get_package_root(__file__))

PACKAGE_ROOT: Path = _pkg.root
"""GNN 包根目录（绝对路径）—— ``packages/gnn/``"""

DATA_DIR: Path = _pkg.data_dir
"""GNN 数据集根目录：``<PACKAGE_ROOT>/data/``"""

OUTPUT_DIR: Path = _pkg.output_dir
"""GNN 实验输出根目录：``<PACKAGE_ROOT>/outputs/``"""

CONFIG_DIR: Path = _pkg.config_dir
"""GNN 配置文件目录：``<PACKAGE_ROOT>/config/``"""


# Path resolution


def resolve_path(path: str | Path) -> Path:
    """解析路径：绝对路径原样返回，相对路径相对于 PACKAGE_ROOT 解析。

    兼容旧 monorepo 前缀 ``"packages/gnn/"``——自动剥离。
    """
    p = Path(path)
    if p.is_absolute():
        return p

    # 兼容旧 monorepo 前缀
    s = str(p).replace("\\", "/")
    legacy_prefix = "packages/gnn/"
    if s.startswith(legacy_prefix):
        p = Path(s[len(legacy_prefix):])
    elif s == "packages/gnn":
        p = Path(".")

    return _pkg.resolve(p)


def resolve_config(path: str | Path) -> Path:
    """解析配置文件路径：优先相对于 PACKAGE_ROOT 查找，其次 CWD。

    这解决了从不同 CWD 启动时 ``--config config/xxx.yaml`` 找不到文件的问题。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    # 先尝试相对于包根目录（用户通常输入 "config/xxx.yaml"）
    candidate = _pkg.resolve(p)
    if candidate.exists():
        return candidate
    # 回退到 CWD
    return (Path.cwd() / p).resolve()
