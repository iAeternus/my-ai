from dataclasses import dataclass
from pathlib import Path

DATA_SET_PATH = "packages\\iris_logistic_regression\\data\\iris"
OUTPUT_PATH = "packages\\iris_logistic_regression\\experiments"


@dataclass(slots=True)
class IrisConfig:
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
    save_dir: Path = Path(OUTPUT_PATH)
    save_name: str = "best_model.pt"

    # Runtime
    compile_model: bool = False
