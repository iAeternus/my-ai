"""数据集下载与缓存工具"""

from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from pathlib import Path


def download_file(
    url: str,
    dest: Path,
    *,
    desc: str = "",
) -> None:
    """下载文件到本地路径

    Args:
        url: 下载地址
        dest: 目标文件路径（包含文件名）
        desc: 可选的下载描述（用于日志）
    """
    label = desc or url.rsplit("/", 1)[-1]
    print(f"下载 {label} -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  完成。")


def extract_tar(
    path: Path,
    dest: Path,
    *,
    mode: str = "r:gz",
) -> None:
    """解压 tar 归档

    Args:
        path: tar 文件路径
        dest: 解压目标目录
        mode: tar 打开模式（默认 ``"r:gz"`` 对应 .tar.gz）
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode) as tar:
        tar.extractall(path=dest)


def extract_zip(path: Path, dest: Path) -> None:
    """解压 zip 归档

    Args:
        path: zip 文件路径
        dest: 解压目标目录
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest)
