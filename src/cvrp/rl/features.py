from typing import Dict, List, Set

from src.cvrp.data.instance import CVRPInstance


def visited_mask_int(instance: CVRPInstance, visited_customers: Set[int]) -> int:
    """
    Convertit l'ensemble des clients visités en masque entier.

    Utile pour Q-learning tabulaire :
    état = (noeud courant, capacité restante, masque_visités, routes_utilisées)
    """

    mask = 0

    for bit_index, customer in enumerate(instance.customers):
        if customer in visited_customers:
            mask |= 1 << bit_index

    return mask


def get_valid_action_mask(
    instance: CVRPInstance,
    current_node: int,
    remaining_capacity: int,
    visited_customers: Set[int],
    current_route: List[int],
    closed_routes: List[List[int]],
    max_routes: int | None = None,
) -> Dict[int, bool]:
    """
    Retourne un masque d'actions valides pour le CVRP.

    Actions possibles :
    - choisir un client non visité dont la demande respecte la capacité restante ;
    - retourner au dépôt si le véhicule est actuellement en route.

    Le dépôt n'est pas autorisé si le véhicule est déjà au dépôt.
    """

    depot = instance.depot

    if max_routes is None:
        max_routes = instance.vehicle_count

    mask = {node: False for node in instance.nodes}

    routes_already_closed = len(closed_routes)

    # Action dépôt : autorisée seulement si on est loin du dépôt
    # et si la route courante contient au moins un client.
    if current_node != depot and len(current_route) > 1:
        mask[depot] = True

    # Si tous les clients sont déjà visités, seul le retour dépôt est utile.
    if len(visited_customers) == instance.customer_count:
        return mask

    # Si on est au dépôt et que le nombre de routes disponibles est déjà consommé,
    # on ne peut pas démarrer une nouvelle route.
    if current_node == depot and max_routes is not None:
        if routes_already_closed >= max_routes:
            return mask

    # Actions clients.
    for customer in instance.customers:
        if customer in visited_customers:
            continue

        demand = instance.demands[customer]

        if demand <= remaining_capacity:
            mask[customer] = True

    return mask


def get_valid_actions(
    instance: CVRPInstance,
    current_node: int,
    remaining_capacity: int,
    visited_customers: Set[int],
    current_route: List[int],
    closed_routes: List[List[int]],
    max_routes: int | None = None,
) -> List[int]:
    """
    Retourne la liste des actions valides à partir du masque.
    """

    mask = get_valid_action_mask(
        instance=instance,
        current_node=current_node,
        remaining_capacity=remaining_capacity,
        visited_customers=visited_customers,
        current_route=current_route,
        closed_routes=closed_routes,
        max_routes=max_routes,
    )

    return [node for node, is_valid in mask.items() if is_valid]


def build_q_learning_state(
    instance: CVRPInstance,
    current_node: int,
    remaining_capacity: int,
    visited_customers: Set[int],
    closed_routes: List[List[int]],
) -> tuple:
    """
    Construit un état discret pour Q-learning.

    Éléments :
    - noeud courant ;
    - capacité restante ;
    - masque entier des clients visités ;
    - nombre de routes déjà fermées.
    """

    return (
        current_node,
        remaining_capacity,
        visited_mask_int(instance, visited_customers),
        len(closed_routes),
    )


def build_node_features(
    instance: CVRPInstance,
    current_node: int,
    remaining_capacity: int,
    visited_customers: Set[int],
    current_route: List[int],
    closed_routes: List[List[int]],
    max_routes: int | None = None,
) -> List[List[float]]:
    """
    Construit les caractéristiques des noeuds pour les modèles neuronaux.

    Ces features serviront surtout pour POMO-style Active Search.

    Features par noeud :
    1. x normalisé ;
    2. y normalisé ;
    3. demande normalisée ;
    4. est dépôt ;
    5. est noeud courant ;
    6. est déjà visité ;
    7. ratio capacité restante ;
    8. ratio routes utilisées ;
    9. action faisable.
    """

    xs = [coord[0] for coord in instance.coordinates.values()]
    ys = [coord[1] for coord in instance.coordinates.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    if max_routes is None:
        max_routes = instance.vehicle_count or 1

    action_mask = get_valid_action_mask(
        instance=instance,
        current_node=current_node,
        remaining_capacity=remaining_capacity,
        visited_customers=visited_customers,
        current_route=current_route,
        closed_routes=closed_routes,
        max_routes=max_routes,
    )

    remaining_capacity_ratio = remaining_capacity / instance.capacity
    routes_used_ratio = len(closed_routes) / max(max_routes, 1)

    features = []

    for node in instance.nodes:
        x, y = instance.coordinates[node]

        x_norm = (x - min_x) / range_x
        y_norm = (y - min_y) / range_y
        demand_norm = instance.demands.get(node, 0) / instance.capacity

        is_depot = 1.0 if node == instance.depot else 0.0
        is_current = 1.0 if node == current_node else 0.0
        is_visited = 1.0 if node in visited_customers else 0.0
        is_feasible = 1.0 if action_mask.get(node, False) else 0.0

        features.append(
            [
                x_norm,
                y_norm,
                demand_norm,
                is_depot,
                is_current,
                is_visited,
                remaining_capacity_ratio,
                routes_used_ratio,
                is_feasible,
            ]
        )

    return features