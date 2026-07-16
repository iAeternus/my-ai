from typing import cast

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

PLANETOID_DATASETS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def load_planetoid(name: str, root: str = "data") -> Data:
    if name not in PLANETOID_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {name}. Available: {list(PLANETOID_DATASETS)}"
        )

    dataset = Planetoid(root=root, name=PLANETOID_DATASETS[name])
    return cast(Data, dataset[0])
