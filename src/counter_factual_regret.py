import numpy as np

from regret_matching import RegretMatching
from util import get_nodes_top_down, linear_utility


class CounterFactualRegret:
    def __init__(self, tfsdp: dict, plus: bool = False):
        """Initialize CFR from a TFSDP description.

        Args:
            tfsdp: dict
            plus: bool
                Whether to use regret matching+ (RM+) or vanilla regret matching (RM).
        """
        self._J = tfsdp["J"]
        self._A = tfsdp["A"]
        self._K = tfsdp["K"]
        self._S = tfsdp["S"]
        self._rho = tfsdp["rho"]
        self._Sigma = tfsdp["Sigma"]
        self._p = tfsdp["p"]

        self._J_set = set(self._J)
        self._K_set = set(self._K)
        self._nodes = get_nodes_top_down(tfsdp)

        self._rms = {j: RegretMatching(len(self._A[j]), plus=plus) for j in self._J}
        self._local_strats = {j: None for j in self._J}

    def next_strategy(self) -> dict:
        """Return current strategy x.

        Returns:
            x: dict[sequence -> [0, 1]]
                current strategy for each sequence. (sequence-form)
        """
        local_strats = {j: self._rms[j].next_strategy() for j in self._J}
        self._local_strats = local_strats  # will be used in observe_utility

        x = {sigma: 0.0 for sigma in self._Sigma}
        for j in self._J:
            parent = self._p[j]
            parent_prob = 1.0 if parent is None else x[parent]
            for idx, a in enumerate(self._A[j]):
                x[(j, a)] = parent_prob * local_strats[j][idx]

        return x

    def observe_utility(self, l: dict):
        """Observe utility l.

        Args:
            l: dict[sequence -> float]
                Linear utility for each sequence.
        """
        V = {}
        V["T"] = 0.0
        for node in reversed(self._nodes):
            if node in self._J_set:
                local = self._local_strats[node]
                V[node] = sum(
                    local[idx] * (l[(node, a)] + V[self._rho[(node, a)]])
                    for idx, a in enumerate(self._A[node])
                )
            elif node in self._K_set:
                V[node] = sum(V[self._rho[(node, s)]] for s in self._S[node])

        for j in self._J:
            l_j = np.zeros(len(self._A[j]))
            for idx, a in enumerate(self._A[j]):
                l_j[idx] = l[(j, a)] + V[self._rho[(j, a)]]
            self._rms[j].observe_utility(l_j)

    def average_strategy(self) -> dict:
        """Compute the average strategy x_bar. Only called at the end of training.

        Returns:
            x_bar: dict[sequence -> [0, 1]]
                final average strategy for each sequence. (sequence-form)
        """
        local_average = {j: self._rms[j].average_strategy() for j in self._J}

        x_bar = {sigma: 0.0 for sigma in self._Sigma}
        for j in self._J:
            parent = self._p[j]
            parent_prob = 1.0 if parent is None else x_bar[parent]
            for idx, a in enumerate(self._A[j]):
                x_bar[(j, a)] = parent_prob * local_average[j][idx]

        return x_bar


class CounterFactualRegretTrainer:
    """CFRM trainer for 2-player zero-sum extensive_form_games.
    One CFRM object for each player.
    """

    def __init__(self, efg: dict, tfsdp0: dict, tfsdp1: dict, plus: bool = False):
        """

        Args:
            efg: dict
                EFG representation of the game. Used for utility computation.
            tfsdp0: dict
                TFSDP representation for player 0. Used for CFRM training.
            tfsdp1: dict
                TFSDP representation for player 1. Used for CFRM training.
            plus: bool
                Whether to use regret matching+ (RM+) or vanilla regret matching (RM).
        """
        self._efg = efg
        self._tfsdp0 = tfsdp0
        self._tfsdp1 = tfsdp1

        self._cfr0 = CounterFactualRegret(self._tfsdp0, plus)
        self._cfr1 = CounterFactualRegret(self._tfsdp1, plus)

    def _compute_utility(self, x0: dict, x1: dict) -> tuple[dict, dict]:
        """
        Compute linear utility for each player given their strategies.

        Args:
            x0: dict[sequence -> [0, 1]]
                strategy of player 0. (sequence-form)
            x1: dict[sequence -> [0, 1]]
                strategy of player 1. (sequence-form)

        Returns:
            l0: dict[sequence -> float]
                linear utility for player 0 for each sequence.
            l1: dict[sequence -> float]
                linear utility for player 1 for each sequence.
        """
        l0 = linear_utility(self._efg, self._tfsdp0, player=0, opponent_x=x1)
        l1 = linear_utility(self._efg, self._tfsdp1, player=1, opponent_x=x0)
        return l0, l1

    def step(self):
        """One iteration of CFRM training.

        Returns:
            x_bar_0: dict[sequence -> [0, 1]]
                final trained strategy of player 0. (sequence-form)
            x_bar_1: dict[sequence -> [0, 1]]
                final trained strategy of player 1. (sequence-form)
        """
        x0 = self._cfr0.next_strategy()
        x1 = self._cfr1.next_strategy()

        l0, l1 = self._compute_utility(x0, x1)

        self._cfr0.observe_utility(l0)
        self._cfr1.observe_utility(l1)
    
    def average_strategy(self) -> tuple[dict, dict]:
        """Compute the average strategy for both players. Only called at the end of training.

        Returns:
            x_bar_0: dict[sequence -> [0, 1]]
                final average strategy of player 0. (sequence-form)
            x_bar_1: dict[sequence -> [0, 1]]
                final average strategy of player 1. (sequence-form)
        """
        x_bar_0 = self._cfr0.average_strategy()
        x_bar_1 = self._cfr1.average_strategy()
        return x_bar_0, x_bar_1
