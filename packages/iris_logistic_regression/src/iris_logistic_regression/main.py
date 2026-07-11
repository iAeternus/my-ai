import logging
from pathlib import Path
from iris_logistic_regression.config import IrisConfig
from iris_logistic_regression.dataset import ID_TO_LABEL, build_dataloader, load_iris
from iris_logistic_regression.model import IrisClassifier
from iris_logistic_regression.parser import parse_args
from iris_logistic_regression.plots import plot_train_history
from iris_logistic_regression.trainer import IrisTrainer
from iris_logistic_regression.utils.device import get_device
from iris_logistic_regression.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = IrisConfig()

    device = get_device()
    features, labels = load_iris(cfg.data_set_dir)
    train_loader, val_loader, test_loader = build_dataloader(
        features, labels, cfg, device
    )
    model = IrisClassifier()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数数量: %d", num_params)

    trainer = IrisTrainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        device=device,
    )

    if args.train:
        history = trainer.train()
        plot_train_history(
            history, save=True, save_dir=cfg.output_dir / "history.png"
        )
    else:
        trainer.load_checkpoint(cfg.output_dir / cfg.save_name)
        sample = features[0]
        pred, prob = trainer.predict_one(sample)

        logger.info("Prediction: %s", ID_TO_LABEL[pred])
        logger.info("Probability: %s", prob.tolist())


if __name__ == "__main__":
    main()
