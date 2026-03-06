from collections import deque

import numpy as np

try:
    from regret_matching import RegretMatching
except ModuleNotFoundError:
    from .regret_matching import RegretMatching


class CounterFactualRegret:
    def __init__(self, tfsdp: dict):
        """Initialize CFR from a TFSDP description.

        Args:
            tfsdp: dict
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
        self._nodes = self._topological_order()

        self._rms = {j: RegretMatching(len(self._A[j])) for j in self._J}
        self._local_strats = {j: None for j in self._J}

    def _topological_order(self) -> list:
        """Return a stable topological order over the all nodes in J and K."""
        nodes = [*self._K, *self._J]
        indegree = {node: 0 for node in nodes}
        children = {node: [] for node in nodes}

        for k in self._K:
            for s in self._S[k]:
                child = self._rho[(k, s)]
                if child in indegree:
                    indegree[child] += 1
                    children[k].append(child)

        for j in self._J:
            for a in self._A[j]:
                child = self._rho[(j, a)]
                if child in indegree:
                    indegree[child] += 1
                    children[j].append(child)

        queue = deque(node for node in nodes if indegree[node] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        return order

    def next_strategy(self) -> dict:
        """Return current strategy x. 

        Returns:
            x: Sigma -> [0, 1] (sequence-form)
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
            x_bar: Sigma -> [0, 1] (sequence-form)
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
            efg: dict
                EFG representation of the game. Used for utility computation.
            tfsdp0: dict
                TFSDP representation for player 0. Used for CFRM training.
            tfsdp1: dict
                TFSDP representation for player 1. Used for CFRM training.
        """
        self._efg = efg
        self._tfsdp0 = tfsdp0
        self._tfsdp1 = tfsdp1

        self._cfr0 = CounterFactualRegret(self._tfsdp0)
        self._cfr1 = CounterFactualRegret(self._tfsdp1)

    def _traverse_tree(
        self,
        node_id: str,
        x0: dict,
        x1: dict,
        l0: dict,
        l1: dict,
        seq0=None,
        seq1=None,
        chance_prob: float = 1.0,
    ) -> None:
        """Traverse the EFG and accumulate sequence-form utilities."""
        node = self._efg[node_id]
        node_type = node["type"]

        if node_type == "CHANCE":
            for _, next_node, prob in node["outcomes"]:
                self._traverse_tree(next_node, x0, x1, l0, l1, seq0, seq1, chance_prob * prob)
            return

        if node_type == "DECISION":
            player = node["player"]
            info_set = node["information_set"]
            for action, next_node in node["actions"]:
                next_seq = (info_set, action)
                if player == 0:
                    self._traverse_tree(next_node, x0, x1, l0, l1, next_seq, seq1, chance_prob)
                else:
                    self._traverse_tree(next_node, x0, x1, l0, l1, seq0, next_seq, chance_prob)
            return

        if node_type == "TERMINAL":
            # accumulatte utility
            if seq0 is not None:
                opp_reach = 1.0 if seq1 is None else x1[seq1]
                l0[seq0] += chance_prob * opp_reach * node["utility"][0]
            if seq1 is not None:
                opp_reach = 1.0 if seq0 is None else x0[seq0]
                l1[seq1] += chance_prob * opp_reach * node["utility"][1]
            return

    def _compute_utility(self, x0: dict, x1: dict) -> tuple[dict, dict]:
        """
        Use EFG to compute utility for each player given their strategies.

        Args:
            x0: Sigma -> [0, 1], strategy of player 0. (sequence-form)
            x1: Sigma -> [0, 1], strategy of player 1. (sequence-form)

        Returns:
            l0: Sigma -> R, utility for player 0 for each sequence.
            l1: Sigma -> R, utility for player 1 for each sequence.
        """
        l0 = {sigma: 0.0 for sigma in self._tfsdp0["Sigma"]}
        l1 = {sigma: 0.0 for sigma in self._tfsdp1["Sigma"]}
        self._traverse_tree("", x0, x1, l0, l1)
        return l0, l1

    def train(self, T: int):
        """Train CFRM for T iterations and return the average strategy.
        
        Returns:
            x_bar_0: Sigma -> [0, 1], final trained strategy of player 0. (sequence-form)
            x_bar_1: Sigma -> [0, 1], final trained strategy of player 1. (sequence-form)
        """
        for t in range(T):
            x0 = self._cfr0.next_strategy()
            x1 = self._cfr1.next_strategy()

            l0, l1 = self._compute_utility(x0, x1)

            self._cfr0.observe_utility(l0)
            self._cfr1.observe_utility(l1)

        x_bar_0 = self._cfr0.average_strategy()
        x_bar_1 = self._cfr1.average_strategy()
        return x_bar_0, x_bar_1
