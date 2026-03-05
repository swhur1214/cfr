import numpy as np

from regret_minimizer import RegretMatching


class CounterFactualRegret:
    def __init__(self, J: list, A: dict, K: list, S: dict, rho: dict, Sigma: list, p: dict):
        """CFR implementation with basic components.
        
        Args:
            J: Decision points in top-down order.
                e.g., ["K", "K|check-bet", ...]
            A[j]: Actions at decision point j.
                e.g., A["K|check-bet"] = ["call", "fold"]
            K: Observation nodes.
                e.g., ["", "K|check", "K|check-check", ...]
            S[k]: Signals at observation node k.
                e.g., S["K|check-check"] = "J"
            rho[(j, a)] or rho[(k, s)]: next node after action/signal.
                e.g., rho[("K|check", "check")] = "K|check-check"
            Sigma: A list of all sequences.
                e.g., Sigma = [("K", "check"), ("K", "bet"), ...]
            p[j]: Parent sequence of decision point.
                e.g., p["K|check-bet"] = ("K", "check")
        """
        self._regret_minimizers = {j: RegretMatching(len(A[j])) for j in J}
        self._local_strategies = {j: None for j in J}

    def next_strategy(self) -> dict:
        """Return current strategy x. 

        Returns:
            x: Sigma -> [0, 1]
                Note: using sequence-form representation instead of behavioral strategy.
        """
        local_strategies = {j: self._regret_minimizers[j].next_strategy() for j in self.J}
        self._local_strategies = local_strategies # will be used in observe_utility

        x = {sigma: 0.0 for sigma in self.Sigma}

        for j in self.J:
            parent = self.p[j]
            parent_prob = 1.0 if parent is None else x[parent]
            for idx, a in enumerate(self.A[j]):
                x[(j, a)] = parent_prob * local_strategies[j][idx]

        return x

    def observe_utility(self, l: dict):
        """Observe utility l.

        Args:
            l: Sigma -> R. Linear utility for each sequence.
        """
        # TODO: Implement V computation
        V = {}
        for node in reversed(self.nodes):
            if node in self.J:
                # V[node] = 
                pass
            if node in self.K:
                # V[node] = 
                pass


        for j in self.J:
            l_j = np.zeros(len(self.A[j]))
            for idx, a in enumerate(self.A[j]):
                l_j[idx] = l[(j, a)] + V[self.rho[(j, a)]]
            self._regret_minimizers[j].observe_utility(l_j)

    def average_strategy(self) -> dict:
        """Compute the average strategy x_bar. Only called at the end of training.

        Returns:
            x_bar: Sigma -> [0, 1]
                Note: using sequence-form representation instead of behavioral strategy.
        """
        local_average = {
            j: self._regret_minimizers[j].average_strategy() for j in self.J
        }
        x_bar = {sigma: 0.0 for sigma in self.Sigma}
        for j in self.J:
            parent = self.p[j]
            parent_prob = 1.0 if parent is None else x_bar[parent]
            for idx, a in enumerate(self.A[j]):
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
        
        
        return l0, l1

    def train(self, T: int):
        """Train CFRM for T iterations and return the average strategy.
        
        Returns:
            x_bar_0: Sigma -> [0, 1], average strategy of player 0. (sequence-form representation)
            x_bar_1: Sigma -> [0, 1], average strategy of player 1. (sequence-form representation)
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


# Backward compatibility for existing imports/usages.
CounterFactualRegretMinimizer = CounterFactualRegret
CounterFactualRegretMinimizerTrainer = CounterFactualRegretTrainer
    
