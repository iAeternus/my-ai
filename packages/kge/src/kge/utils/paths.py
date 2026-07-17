"""路径解析基础设施（委托给 core.utils.paths）。

通过 core 的 ``get_package_root()`` 定位 KGE 包根目录，
``PROJECT_ROOT`` 从 core 透传（monorepo 根）。
"""

from __future__ import annotations

from pathlib import Path

from core.utils.paths import PROJECT_ROOT, PackagePaths, get_package_root

# ── KGE 包级路径 ───────────────────────────────────────────────────────

_pkg = PackagePaths(get_package_root(__file__))

PACKAGE_ROOT: Path = _pkg.root
"""KGE 包根目录（绝对路径）—— ``packages/kge/``"""

DATA_DIR: Path = _pkg.data_dir
"""KGE 数据集根目录：``<PACKAGE_ROOT>/data/``"""

OUTPUT_DIR: Path = _pkg.output_dir
"""KGE 实验输出根目录：``<PACKAGE_ROOT>/outputs/``"""


# ── 路径解析（向后兼容）─────────────────────────────────────────────────


def resolve_path(path: str | Path) -> Path:
    """将路径解析为绝对路径。

    规则：
    - 绝对路径 → 原样返回。
    - ``"packages/kge/..."`` → 剥离旧 monorepo 前缀，相对于 ``PACKAGE_ROOT`` 解析。
    - 其他相对路径 → 相对于 ``PACKAGE_ROOT`` 解析。
    """
    p = Path(path)
    if p.is_absolute():
        return p

    # [legacy] 兼容旧 monorepo 前缀
    s = str(p).replace("\\", "/")
    legacy_prefix = "packages/kge/"
    if s.startswith(legacy_prefix):
        p = Path(s[len(legacy_prefix):])
    elif s == "packages/kge":
        p = Path(".")

    return _pkg.resolve(p)
