import logging

logger = logging.getLogger("graph_learning")


class EarlyStopping:
    """早停

    Args:
        patience: 无改善的容忍轮数，超过后停止训练
        min_delta: 监控指标的最小变化量，低于此值不视为改善
        mode: ``"max"``（越高越好）或 ``"min"``（越低越好）
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Mode must be `max` or `min`")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float | None = None
        self._counter: int = 0

    @property
    def stopped(self) -> bool:
        """是否应停止训练"""
        return self._counter >= self.patience

    def step(self, value: float) -> bool:
        """用新的指标值更新追踪器

        Args:
            value: 最新指标值

        Returns:
            若应停止训练则返回 ``True``，否则返回 ``False``
        """
        if self._best is None:
            self._best = value
            return False

        improved: bool
        if self.mode == "max":
            improved = value > self._best + self.min_delta
        else:
            improved = value < self._best - self.min_delta

        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            logger.info(
                "早停: %d / %d (最佳=%.4f)",
                self._counter,
                self.patience,
                self._best,
            )

        return self.stopped
