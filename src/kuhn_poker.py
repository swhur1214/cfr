class KuhnPoker:
    """Implementation of Kuhn Poker game and its representations (EFG and TFSDP)."""

    CARDS = ("J", "Q", "K")
    RANK = {"J": 0, "Q": 1, "K": 2}
    DEALS = ["KQ", "QK", "JK", "KJ", "QJ", "JQ"]

    @staticmethod
    def showdown_utility(card0: str, card1: str, amount: int) -> list[int]:
        u0 = amount if KuhnPoker.RANK[card0] > KuhnPoker.RANK[card1] else -amount
        return [u0, -u0]

    @staticmethod
    def efg() -> dict:
        """Return Kuhn Poker as an extensive-form game.

        Returns:
            efg: dict
                Node map keyed by history strings such as `""`, `KQ`, or `KQ|check-bet`.
                Node formats are:
                `{"type": "CHANCE", "outcomes": [(outcome, next_node, prob), ...]}`
                `{"type": "DECISION", "player": int, "information_set": str,
                "actions": [(action, next_node), ...]}`
                `{"type": "TERMINAL", "utility": [u0, u1]}`
        """
        efg = {
            "": {
                "type": "CHANCE",
                "outcomes": [],
            }
        }

        for deal in KuhnPoker.DEALS:
            card0, card1 = deal
            efg[""]["outcomes"].append(
                (deal, deal, 1 / 6)
            )  # (outcome, next_node, probability)

            # Player 0 first action (check / bet)
            efg[deal] = {
                "type": "DECISION",
                "player": 0,
                "information_set": card0,
                "actions": [
                    ("check", f"{deal}|check"),
                    ("bet", f"{deal}|bet"),
                ],  # (action, next_node)
            }

            # After check, player 1 can check or bet
            efg[f"{deal}|check"] = {
                "type": "DECISION",
                "player": 1,
                "information_set": f"{card1}|check",
                "actions": [
                    ("check", f"{deal}|check-check"),
                    ("bet", f"{deal}|check-bet"),
                ],
            }

            # After check-bet, player 0 can call or fold
            efg[f"{deal}|check-bet"] = {
                "type": "DECISION",
                "player": 0,
                "information_set": f"{card0}|check-bet",
                "actions": [
                    ("call", f"{deal}|check-bet-call"),
                    ("fold", f"{deal}|check-bet-fold"),
                ],
            }

            # After bet, player 1 can call or fold
            efg[f"{deal}|bet"] = {
                "type": "DECISION",
                "player": 1,
                "information_set": f"{card1}|bet",
                "actions": [("call", f"{deal}|bet-call"), ("fold", f"{deal}|bet-fold")],
            }

            # Terminal nodes
            efg[f"{deal}|check-check"] = {
                "type": "TERMINAL",
                "utility": KuhnPoker.showdown_utility(card0, card1, amount=1),
            }
            efg[f"{deal}|check-bet-call"] = {
                "type": "TERMINAL",
                "utility": KuhnPoker.showdown_utility(card0, card1, amount=2),
            }
            efg[f"{deal}|check-bet-fold"] = {
                "type": "TERMINAL",
                "utility": [-1, 1],
            }
            efg[f"{deal}|bet-call"] = {
                "type": "TERMINAL",
                "utility": KuhnPoker.showdown_utility(card0, card1, amount=2),
            }
            efg[f"{deal}|bet-fold"] = {
                "type": "TERMINAL",
                "utility": [1, -1],
            }

        return efg

    @staticmethod
    def tfsdp(player: int) -> dict:
        """Return the player's TFSDP representation.

        Args:
            player: Player index (`0` or `1`).

        Returns:
            tfsdp: dict
                Mapping with TFSDP components:
                J: decision points.
                A: legal actions A_j for each j in J.
                K: observation points.
                S: possible signals S_k for each k in K.
                rho: transition map from (j, a) or (k, s) to the next point.
                Sigma: set of sequences (j, a).
                p: parent sequence p_j of each decision point j.
        """
        if player == 0:
            J = [*KuhnPoker.CARDS, *[f"{c}|check-bet" for c in KuhnPoker.CARDS]]
            A = {c: ["check", "bet"] for c in KuhnPoker.CARDS}
            A.update({f"{c}|check-bet": ["call", "fold"] for c in KuhnPoker.CARDS})
            p = {c: None for c in KuhnPoker.CARDS}
            p.update({f"{c}|check-bet": (c, "check") for c in KuhnPoker.CARDS})

            K = ["", *[f"{c}|check" for c in KuhnPoker.CARDS]]
            S = {
                "": list(KuhnPoker.CARDS),
                **{f"{c}|check": ["check", "bet"] for c in KuhnPoker.CARDS},
            }

            rho = {}
            for c in KuhnPoker.CARDS:
                rho[("", c)] = c
                rho[(c, "check")] = f"{c}|check"
                rho[(c, "bet")] = "T"
                rho[(f"{c}|check", "check")] = "T"
                rho[(f"{c}|check", "bet")] = f"{c}|check-bet"
                rho[(f"{c}|check-bet", "call")] = "T"
                rho[(f"{c}|check-bet", "fold")] = "T"

        else:
            J = [f"{c}|check" for c in KuhnPoker.CARDS] + [
                f"{c}|bet" for c in KuhnPoker.CARDS
            ]
            A = {f"{c}|check": ["check", "bet"] for c in KuhnPoker.CARDS}
            A.update({f"{c}|bet": ["call", "fold"] for c in KuhnPoker.CARDS})
            p = {j: None for j in J}

            K = ["", *KuhnPoker.CARDS]
            S = {
                "": list(KuhnPoker.CARDS),
                **{c: ["check", "bet"] for c in KuhnPoker.CARDS},
            }

            rho = {}
            for c in KuhnPoker.CARDS:
                rho[("", c)] = c
                rho[(c, "check")] = f"{c}|check"
                rho[(c, "bet")] = f"{c}|bet"
                rho[(f"{c}|check", "check")] = "T"
                rho[(f"{c}|check", "bet")] = "T"
                rho[(f"{c}|bet", "call")] = "T"
                rho[(f"{c}|bet", "fold")] = "T"

        Sigma = [(j, a) for j in J for a in A[j]]

        return {
            "J": J,
            "A": A,
            "K": K,
            "S": S,
            "rho": rho,
            "Sigma": Sigma,
            "p": p,
        }
