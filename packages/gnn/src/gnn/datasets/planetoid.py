from torch_geometric.datasets import Planetoid

PLANETOID_DATASETS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def load_planetoid(name: str, root: str = "data"):
    if name not in PLANETOID_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {name}. " f"Available: {list(PLANETOID_DATASETS)}"
        )

    return Planetoid(root=root, name=PLANETOID_DATASETS[name])
