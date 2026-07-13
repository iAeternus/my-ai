from gnn.config.schema import Config, TaskType
from gnn.trainer.node_cls import NodeClassificationTrainer
from gnn.trainer.link_pred import LinkPredictionTrainer


def create_trainer(cfg: Config, model, device):
    if cfg.task == TaskType.NODE_CLASSIFICATION:
        return NodeClassificationTrainer(cfg, model, device)
    elif cfg.task == TaskType.LINK_PREDICTION:
        return LinkPredictionTrainer(cfg, model, device)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")
