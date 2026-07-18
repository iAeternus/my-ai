from dataclasses import Field, asdict
import json
from pathlib import Path
from typing import Any, ClassVar, Protocol


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class SerializableConfig(DataclassInstance):
    """配置序列化 mixin，提供 ``to_dict`` / ``to_json`` 默认实现

    各包的 ``Config`` 类继承此类即可自动获得序列化能力
    """

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典"""
        return asdict(self)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """保存配置为 JSON 文件

        Args:
            path: 输出文件路径
            indent: JSON 缩进空格数
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
