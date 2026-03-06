from collections import deque


def get_nodes_top_down(tfsdp: dict) -> list:
    """Return all TFSDP nodes in a top-down order."""
    nodes = [*tfsdp["K"], *tfsdp["J"]]
    indegree = {node: 0 for node in nodes}
    children = {node: [] for node in nodes}

    for obs_point in tfsdp["K"]:
        for signal in tfsdp["S"][obs_point]:
            child = tfsdp["rho"][(obs_point, signal)]
            if child in indegree:
                indegree[child] += 1
                children[obs_point].append(child)

    for info_set in tfsdp["J"]:
        for action in tfsdp["A"][info_set]:
            child = tfsdp["rho"][(info_set, action)]
            if child in indegree:
                indegree[child] += 1
                children[info_set].append(child)

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


def sequence_to_behavior(x: dict, tfsdp: dict) -> dict:
    """Convert a sequence-form realization plan into a behavioral strategy."""
    behavior = {}

    for info_set in tfsdp["J"]:
        actions = tfsdp["A"][info_set]
        parent = tfsdp["p"][info_set]
        parent_reach = 1.0 if parent is None else float(x[parent])

        if parent_reach > 0.0:
            behavior[info_set] = {
                action: float(x[(info_set, action)] / parent_reach)
                for action in actions
            }
        else:
            uniform = 1.0 / len(actions)
            behavior[info_set] = {action: uniform for action in actions}

    return behavior


def _expected_value_recursive(
    efg: dict,
    pi0: dict,
    pi1: dict,
    node_id: str = "",
) -> float:
    """Return player 0's expected payoff from the given node."""
    node = efg[node_id]
    node_type = node["type"]

    if node_type == "CHANCE":
        return sum(
            prob * _expected_value_recursive(efg, pi0, pi1, next_node)
            for _, next_node, prob in node["outcomes"]
        )

    if node_type == "DECISION":
        info_set = node["information_set"]
        strategy = pi0 if node["player"] == 0 else pi1
        return sum(
            strategy[info_set][action]
            * _expected_value_recursive(efg, pi0, pi1, next_node)
            for action, next_node in node["actions"]
        )

    return float(node["utility"][0])


def expected_value(
    efg: dict,
    tfsdp0: dict,
    tfsdp1: dict,
    x0: dict,
    x1: dict,
    player: int = 0,
) -> float:
    """Return the expected payoff under sequence-form strategies."""
    pi0 = sequence_to_behavior(x0, tfsdp0)
    pi1 = sequence_to_behavior(x1, tfsdp1)
    payoff0 = _expected_value_recursive(efg, pi0, pi1)
    return payoff0 if player == 0 else -payoff0


def _traverse_tree_for_linear_utility(
    efg: dict,
    node_id: str,
    player: int,
    opponent_x: dict,
    l: dict,
    seq_player=None,
    seq_opp=None,
    chance_prob: float = 1.0,
) -> None:
    """Accumulate sequence-form linear utility against a fixed opponent strategy."""
    node = efg[node_id]
    node_type = node["type"]

    if node_type == "CHANCE":
        for _, next_node, prob in node["outcomes"]:
            _traverse_tree_for_linear_utility(
                efg,
                next_node,
                player,
                opponent_x,
                l,
                seq_player,
                seq_opp,
                chance_prob * prob,
            )
        return

    if node_type == "DECISION":
        info_set = node["information_set"]
        acting_player = node["player"]
        for action, next_node in node["actions"]:
            next_seq = (info_set, action)
            if acting_player == player:
                _traverse_tree_for_linear_utility(
                    efg,
                    next_node,
                    player,
                    opponent_x,
                    l,
                    next_seq,
                    seq_opp,
                    chance_prob,
                )
            else:
                _traverse_tree_for_linear_utility(
                    efg,
                    next_node,
                    player,
                    opponent_x,
                    l,
                    seq_player,
                    next_seq,
                    chance_prob,
                )
        return

    if seq_player is not None:
        opp_reach = 1.0 if seq_opp is None else float(opponent_x[seq_opp])
        l[seq_player] += chance_prob * opp_reach * float(node["utility"][player])


def linear_utility(efg: dict, tfsdp: dict, player: int, opponent_x: dict) -> dict:
    """Return the linear utility vector for the player's sequences."""
    l = {sigma: 0.0 for sigma in tfsdp["Sigma"]}
    _traverse_tree_for_linear_utility(efg, "", player, opponent_x, l)
    return l


def best_response_value(efg: dict, tfsdp: dict, player: int, opponent_x: dict) -> float:
    """Return the value of the player's best response to the opponent strategy."""
    l = linear_utility(efg, tfsdp, player, opponent_x)

    decision_points = set(tfsdp["J"])
    observation_points = set(tfsdp["K"])
    nodes = get_nodes_top_down(tfsdp)

    values = {"T": 0.0}
    for node in reversed(nodes):
        if node in decision_points:
            values[node] = max(
                l[(node, action)] + values[tfsdp["rho"][(node, action)]]
                for action in tfsdp["A"][node]
            )
        elif node in observation_points:
            values[node] = sum(
                values[tfsdp["rho"][(node, signal)]]
                for signal in tfsdp["S"][node]
            )

    return values[""]


def exploitability(efg: dict, tfsdp0: dict, tfsdp1: dict, x0: dict, x1: dict) -> float:
    """Return exploitability, i.e. NashConv / 2, for a 2-player zero-sum game."""
    br0 = best_response_value(efg, tfsdp0, player=0, opponent_x=x1)
    br1 = best_response_value(efg, tfsdp1, player=1, opponent_x=x0)
    return 0.5 * (br0 + br1)


def format_strategy(x: dict, tfsdp: dict, digits: int = 4) -> dict:
    """Return a rounded behavioral strategy by information set."""
    behavior = sequence_to_behavior(x, tfsdp)
    return {
        info_set: {
            action: round(float(behavior[info_set][action]), digits)
            for action in tfsdp["A"][info_set]
        }
        for info_set in tfsdp["J"]
    }
