import numpy as np


class RegretMinimizer:
    def __init__(
        self,
        n_actions: int,
        plus: bool = False,
        init_regret: np.ndarray | None = None,
    ):
        self.n_actions = n_actions
        self.plus = plus
        self.regret = (
            init_regret.copy() if init_regret is not None else np.zeros(n_actions)
        )
        self.strategy_sum = np.zeros(n_actions)
        self._last_strategy = np.full(n_actions, 1.0 / n_actions)

    def next_strategy(self) -> np.ndarray:
        pos = np.maximum(self.regret, 0.0)
        s = pos.sum()
        self._last_strategy = (
            pos / s if s > 0 else np.full(self.n_actions, 1.0 / self.n_actions)
        )
        self.strategy_sum += self._last_strategy
        return self._last_strategy

    def observe_utility(self, action_utilities: np.ndarray) -> None:
        expected = float(self._last_strategy @ action_utilities)
        self.regret += action_utilities - expected
        if self.plus:
            self.regret = np.maximum(self.regret, 0.0)

    def average_strategy(self) -> np.ndarray:
        return self.strategy_sum / self.strategy_sum.sum()
