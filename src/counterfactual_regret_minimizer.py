import numpy as np

from regret_minimizer import RegretMinimizer


def _next_node(rho, node, label):
    if callable(rho):
        return rho(node, label)
    return rho[(node, label)]


def build_bottom_up_order(J, K, A, S, rho):
    J_set = set(J)
    K_set = set(K)
    visited = set()
    order = []

    def children(node):
        if node in J_set:
            return [_next_node(rho, node, action) for action in A[node]]
        if node in K_set:
            return [_next_node(rho, node, signal) for signal in S[node]]
        return []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for child in children(node):
            if child in J_set or child in K_set:
                dfs(child)
        order.append(node)

    for node in list(J) + list(K):
        dfs(node)

    return order


class CounterfactualRegretMinimizer:
    """
    CFR implementation with basic components:
    J, A, K, S, p, rho.

    - J: decision points in top-down order
    - A[j]: actions at decision point j
    - K: observation nodes
    - S[k]: signals at observation node k
    - p[j]: parent sequence of decision point j (or None)
    - rho(node, label): next node after action/signal
    """

    def __init__(
        self,
        J,
        K,
        A,
        S,
        p,
        rho,
        plus=False,
    ):
        self.J = list(J)
        self.K = list(K)
        self.A = A
        self.S = S
        self.p = p
        self.rho = rho

        self._J_set = set(self.J)
        self._K_set = set(self.K)

        self.nodes_bottom_up = build_bottom_up_order(
            self.J,
            self.K,
            self.A,
            self.S,
            self.rho,
        )

        self.regret_minimizers = {
            j: RegretMinimizer(len(self.A[j]), plus=plus) for j in self.J
        }

        self.local_strategy = {}
        self.sequence_strategy = {}

    def _next_node(self, node, label):
        return _next_node(self.rho, node, label)

    def next_strategy(self):
        # Step 1: b_j^t <- R_j.NextStrategy()
        self.local_strategy = {}
        for j in self.J:
            self.local_strategy[j] = self.regret_minimizers[j].next_strategy()

        # Step 2: build x^t
        self.sequence_strategy = {}
        for j in self.J:
            parent_sequence = self.p[j]
            parent_reach = (
                1.0 if parent_sequence is None else self.sequence_strategy[parent_sequence]
            )
            for i, action in enumerate(self.A[j]):
                self.sequence_strategy[(j, action)] = (
                    parent_reach * self.local_strategy[j][i]
                )

        return self.sequence_strategy

    def observe_utility(self, u):
        # Step 1: compute V^t in bottom-up order
        V = {}

        for node in self.nodes_bottom_up:
            if node in self._J_set:
                j = node
                V[j] = 0.0
                for i, action in enumerate(self.A[j]):
                    child = self._next_node(j, action)
                    V[j] += self.local_strategy[j][i] * (
                        u[(j, action)] + V.get(child, 0.0)
                    )
            else:
                k = node
                V[k] = 0.0
                for signal in self.S[k]:
                    child = self._next_node(k, signal)
                    V[k] += V.get(child, 0.0)

        # Step 2: build l_j^t and update each R_j
        for j in self.J:
            local_utility = np.zeros(len(self.A[j]))
            for i, action in enumerate(self.A[j]):
                child = self._next_node(j, action)
                local_utility[i] = u[(j, action)] + V.get(child, 0.0)
            self.regret_minimizers[j].observe_utility(local_utility)
