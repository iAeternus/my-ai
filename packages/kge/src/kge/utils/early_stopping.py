from __future__ import annotations

MONITOR_MODES: dict[str, str] = {
    "val_loss": "min",
    "val_mrr": "max",
    "val_hits@1": "max",
    "val_hits@3": "max",
    "val_hits@10": "max",
    "val_acc": "max",
    "val_auc": "max",
    "val_ap": "max",
}


class EarlyStopping:
    """早停判断器

    Args:
        patience: 容忍的连续无改善轮数
        mode: "min" 或 "max"
        min_delta: 最小改善幅度
    """

    def __init__(
        self,
        patience: int = 30,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', get: {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._counter = 0
        self._best: float | None = None
        self.improved = False

    def step(self, metric: float) -> bool:
        """记录指标值，返回是否应停止

        Returns:
            True 如果 patience 耗尽（应早停）
        """

        self.improved = False
        if self._best is None:
            self._best = metric
            self._counter = 0
            self.improved = True
            return False
        
        if self.mode == "max":
            improved = metric > self._best + self.min_delta
        else:
            improved = metric < self._best - self.min_delta
        
        if improved:
            self._best = metric
            self._counter = 0
            self.improved = True
        else:
            self._counter += 1
        
        return self._counter >= self.patience
    
    @property
    def best(self) -> float | None:
        return self._best