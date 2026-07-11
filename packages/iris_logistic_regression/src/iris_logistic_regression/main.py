import logging
from pathlib import Path
from iris_logistic_regression.config import DATA_SET_PATH, OUTPUT_PATH, IrisConfig
from iris_logistic_regression.dataset import build_dataloader, load_iris
from iris_logistic_regression.model import IrisClassifier
from iris_logistic_regression.plots import plot_train_history
from iris_logistic_regression.trainer import IrisTrainer
from iris_logistic_regression.utils import get_device

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    setup_logging()
    cfg = IrisConfig()

    device = get_device()
    features, labels = load_iris(Path(DATA_SET_PATH))
    train_loader, val_loader, test_loader = build_dataloader(features, labels, cfg, device)
    model = IrisClassifier()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model Parameters: %d", num_params)

    trainer = IrisTrainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        device=device,
    )
    history = trainer.train()

    plot_train_history(history, save=True, save_dir=Path(OUTPUT_PATH) / "history.png")


if __name__ == "__main__":
    main()
