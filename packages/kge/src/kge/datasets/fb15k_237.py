from __future__ import annotations
import os
from pathlib import Path
import urllib.request
import torch
from torch import Tensor

from kge.utils.paths import DATA_DIR
from kge.datasets.base import BaseKGDataset
from kge.datasets.registry import KGDatasetRegistry

FB15K237_URL = (
    "https://raw.githubusercontent.com/TimDettmers/ConvE/master/FB15k-237.tar.gz"
)


@KGDatasetRegistry.register("fb15k-237")
class FB15k237Dataset(BaseKGDataset):
    """FB15k-237: 14,541 entities, 237 relations, 272K train, 17K valid, 20K test"""

    name = "fb15k-237"

    def __init__(self, root: str = str(DATA_DIR)) -> None:
        super().__init__(root)
        self._num_entities = 0
        self._num_relations = 0

    @property
    def num_entities(self) -> int:
        if self._num_entities == 0:
            self._load()
        return self._num_entities

    @property
    def num_relations(self) -> int:
        if self._num_relations == 0:
            self._load()
        return self._num_relations

    def _load(self) -> None:
        data_dir = self.root / "FB15k-237"
        if not data_dir.exists():
            self._download(data_dir)

        # 构建 ID 映射
        self._entity_to_id, self._relation_to_id = {}, {}
        self._id_to_entity, self._id_to_relation = {}, {}

        def _register_entity(e: str) -> int:
            if e not in self._entity_to_id:
                eid = len(self._entity_to_id)
                self._entity_to_id[e] = eid
                self._id_to_entity[eid] = e
            return self._entity_to_id[e]

        def _register_relation(r: str) -> int:
            if r not in self._relation_to_id:
                rid = len(self._relation_to_id)
                self._relation_to_id[r] = rid
                self._id_to_relation[rid] = r
            return self._relation_to_id[r]

        for split_file in ["train.txt", "valid.txt", "test.txt"]:
            path = data_dir / split_file
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    h, r, t = parts
                    _register_entity(h)
                    _register_entity(t)
                    _register_relation(r)

        self._num_entities = len(self._entity_to_id)
        self._num_relations = len(self._relation_to_id)

        self._train_triples = torch.tensor(
            self._read_triples(
                data_dir / "train.txt",
                self._entity_to_id,
                self._relation_to_id,
            ),
            dtype=torch.long,
        )
        self._val_triples = torch.tensor(
            self._read_triples(
                data_dir / "valid.txt",
                self._entity_to_id,
                self._relation_to_id,
            ),
            dtype=torch.long,
        )
        self._test_triples = torch.tensor(
            self._read_triples(
                data_dir / "test.txt",
                self._entity_to_id,
                self._relation_to_id,
            )
        )

    def _download(self, data_dir: Path) -> None:
        """下载并解压 FB15k-237"""
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / "fb15k-237.tar.gz"

        print(f"下载 FB15k-237 到 {archive_path} ...")
        urllib.request.urlretrieve(FB15K237_URL, archive_path)

        print(f"解压到 {data_dir} ...")
        import tarfile

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

        # tar 解压后可能在子目录中，移动文件
        extracted = data_dir / "FB15k-237"
        if not extracted.is_dir():
            # 查找实际目录
            for item in os.listdir(data_dir):
                full = data_dir / item
                if full.is_dir() and item != "FB15k-237":
                    for f in os.listdir(full):
                        os.rename(full / f, data_dir / f)
                        full.rmdir()
                        break

        os.remove(archive_path)
        print("FB15k-237 下载完成。")
