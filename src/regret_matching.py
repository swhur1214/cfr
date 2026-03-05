import numpy as np


class RegretMatching:
    """Regret Matching (RM / RM+) Algorithm"""

    def __init__(
        self,
        n_actions: int,
        plus: bool = False,
        init_regret: np.ndarray | None = None,
    ):
        self._n_actions = n_actions
        self._plus = plus
        self._regret = (
            init_regret.copy() if init_regret is not None else np.zeros(n_actions)
        )
        self._strategy_sum = np.zeros(n_actions)
        self._last_strategy = np.full(n_actions, 1.0 / n_actions)

    def next_strategy(self) -> np.ndarray:
        """Compute and return the next strategy.

        Returns:
            next_strategy: np.ndarray(n_actions,)
                Probability vector over actions.
        """
        pos = np.maximum(self._regret, 0.0)
        s = pos.sum()
        next_strategy = (
            pos / s if s > 0 else np.full(self._n_actions, 1.0 / self._n_actions)
        )
        self._strategy_sum += next_strategy
        self._last_strategy = next_strategy
        return next_strategy

    def observe_utility(self, l: np.ndarray) -> None:
        """Update regrets from action utilities under the last strategy.

        Args:
            l: np.ndarray(n_actions,)
                Linear utility vector for actions.
        """
        expected = float(self._last_strategy @ l)
        self._regret += l - expected
        if self._plus:
            self._regret = np.maximum(self._regret, 0.0)

    def average_strategy(self) -> np.ndarray:
        return self._strategy_sum / self._strategy_sum.sum()
