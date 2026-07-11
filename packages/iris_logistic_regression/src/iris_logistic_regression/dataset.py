from pathlib import Path
from iris_logistic_regression.config import IrisConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn

LABEL_TO_ID = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}

ID_TO_LABEL = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica",
}


def load_iris(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data_file = data_dir / "bezdekIris.data"
    if not data_file.exists():
        raise FileNotFoundError(f"Iris数据集不存在: {data_file}")

    features: list[list[float]] = []
    labels: list[int] = []

    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            fields = line.split(",")

            features.append(list(map(float, fields[:-1])))
            labels.append(LABEL_TO_ID[fields[-1]])

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def build_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: IrisConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    X = features.numpy()
    y = labels.numpy()

    # train / (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=config.train_set,
        random_state=config.seed,
        stratify=y,
    )

    # val / test
    val_ratio = config.val_set / (
        config.val_set + (1.0 - config.train_set - config.val_set)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_ratio,
        random_state=config.seed,
        stratify=y_temp,
    )

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, val_loader, test_loader
