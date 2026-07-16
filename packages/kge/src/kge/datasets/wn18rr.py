from __future__ import annotations
import os
import tarfile
import urllib.request
from pathlib import Path

import torch
from torch import Tensor

from kge.datasets.base import BaseKGDataset
from kge.datasets.registry import KGDatasetRegistry

WN18RR_URL = "https://raw.githubusercontent.com/TimDettmers/ConvE/master/WN18RR.tar.gz"


@KGDatasetRegistry.register("wn18rr")
class WN18RRDataset(BaseKGDataset):
    """WN18RR: 40,943 entities, 11 relations, 87K train, 3K valid, 3K test"""

    name = "wn18rr"

    def __init__(self, root: str = "packages/kge/data") -> None:
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
        data_dir = self.root / "WN18RR"
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
                data_dir / "train.txt", self._entity_to_id, self._relation_to_id
            ),
            dtype=torch.long,
        )
        self._val_triples = torch.tensor(
            self._read_triples(
                data_dir / "valid.txt", self._entity_to_id, self._relation_to_id
            ),
            dtype=torch.long,
        )
        self._test_triples = torch.tensor(
            self._read_triples(
                data_dir / "test.txt", self._entity_to_id, self._relation_to_id
            ),
            dtype=torch.long,
        )

    def _download(self, data_dir: Path) -> None:
        """下载并解压 WN18RR"""
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / "wn18rr.tar.gz"

        print(f"下载 WN18RR 到 {archive_path} ...")
        urllib.request.urlretrieve(WN18RR_URL, archive_path)

        print(f"解压到 {data_dir} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

        archive_path.unlink()
        print("WN18RR 下载完成")
