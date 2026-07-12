from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IrisConfig:
    # Path
    data_set_dir: Path = Path("packages\\iris_logistic_regression\\data\\iris")
    output_dir: Path = Path("packages\\iris_logistic_regression\\outputs")

    # Dataset
    train_set: float = 0.8
    val_set: float = 0.1
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0

    # Training
    epochs: int = 100
    lr: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 30
    save_name: str = "best_model.pt"

    # Runtime
    compile_model: bool = False
