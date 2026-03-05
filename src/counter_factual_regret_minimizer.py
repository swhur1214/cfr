import numpy as np

from regret_matching import RegretMatching


class CounterFactualRegret:
    def __init__(self, tfsdp: dict):
        """Initialize CFR from a TFSDP description.

        Args:
            tfsdp: 
        """
        self._J = tfsdp["J"]
        self._A = tfsdp["A"]
        self._K = tfsdp["K"]
        self._S = tfsdp["S"]
        self._rho = tfsdp["rho"]
        self._Sigma = tfsdp["Sigma"]
        self._p = tfsdp["p"]

        # TODO: top-down traversal order of J + K.
        # self._nodes = 

        self._rms = {j: RegretMatching(len(self._A[j])) for j in self._J}
        self._local_strats = {j: None for j in self._J}

    def next_strategy(self) -> dict:
        """Return current strategy x. 

        Returns:
            x: Sigma -> [0, 1] ('sequence-form' representation)
        """
        local_strats = {j: self._rms[j].next_strategy() for j in self._J}
        self._local_strats = local_strats # will be used in observe_utility

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
            l: Sigma -> R. Linear utility for each sequence.
        """
        # TODO: Implement V computation
        # Note that V has a non-zero value only at non-terminal nodes.
        # So we can just store V in l.
        for node in reversed(self.nodes):
            if node in self._J:
                # l[node] = 
                pass
            if node in self._K:
                # l[node] = 
                pass


        for j in self._J:
            l_j = np.zeros(len(self._A[j]))
            for idx, a in enumerate(self._A[j]):
                l_j[idx] = l[(j, a)] + V[self._rho[(j, a)]]
            self._rms[j].observe_utility(l_j)

    def average_strategy(self) -> dict:
        """Compute the average strategy x_bar. Only called at the end of training.

        Returns:
            x_bar: Sigma -> [0, 1] ('sequence-form' representation)
        """
        local_average = {
            j: self._rms[j].average_strategy() for j in self._J
        }
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

    def __init__(self, efg: dict, tfsdp0: dict, tfsdp1: dict):
        """

        Args:
            efg: extensive-form game in EFG format.
                e.g., {
                        "": {
                            "type": "CHANCE",
                            "outcomes": [("KQ", "KQ", 1/6), ...], # (outcome, next_node, probability)
                        },
                        "KQ": {
                            "type": "DECISION",
                            "player": 0,
                            "information_set": "K",
                            "actions": [("check", "KQ|check"), ("bet", "KQ|bet")], # (action, next_node)
                        },
                        ...
                        "KQ|check-bet-call": {
                            "type": "TERMINAL",
                            "utility": [2, -2], # utility for player 0 and player 1
                        }
                    }
        """
        self.efg = efg
        self.tfsdp0 = tfsdp0
        self.tfsdp1 = tfsdp1
        self.cfr0 = CounterFactualRegret(**self.tfsdp0)
        self.cfr1 = CounterFactualRegret(**self.tfsdp1)
        

    def compute_utility(self, x0: dict, x1: dict) -> tuple[dict, dict]:
        """
        Use EFG to compute utility for each player given their strategies.

        Args:
            x0: Sigma -> [0, 1], strategy of player 0 in sequence-form representation.
            x1: Sigma -> [0, 1], strategy of player 1 in sequence-form representation.

        Returns:
            l0: Sigma -> R, utility for player 0 for each sequence.
            l1: Sigma -> R, utility for player 1 for each sequence.
        """
        
        # TODO: Implement utility computation using EFG and strategies x0, x1.
        # Note that the utility has a non-zero value only at terminal nodes.
        # i.e. (j, a) such that rho[(j, a)] = "T". 
        return l0, l1

    def train(self, T: int):
        """Train CFRM for T iterations and return the average strategy.
        
        Returns:
            x_bar_0: Sigma -> [0, 1], final trained strategy of player 0. ('sequence-form' representation)
            x_bar_1: Sigma -> [0, 1], final trained strategy of player 1. ('sequence-form' representation)
        """
        for t in range(T):
            x0 = self.cfr0.next_strategy()
            x1 = self.cfr1.next_strategy()

            l0, l1 = self.compute_utility(x0, x1)

            self.cfr0.observe_utility(l0)
            self.cfr1.observe_utility(l1)

        x_bar_0 = self.cfr0.average_strategy()
        x_bar_1 = self.cfr1.average_strategy()
        return x_bar_0, x_bar_1

