from typing import List, Tuple

from src.cvrp.data.instance import CVRPInstance
from src.cvrp.core.validation import normalize_solution


def _route_load(instance: CVRPInstance, route: List[int]) -> int:
    """
    Calcule la charge totale d'une route.
    """
    return instance.route_demand(route)


def _find_route_containing_customer(
    routes: List[List[int]],
    customer: int,
) -> int | None:
    """
    Retourne l'indice de la route qui contient le client donné.
    Le dépôt est ignoré.
    """
    for idx, route in enumerate(routes):
        if customer in route[1:-1]:
            return idx
    return None


def _is_customer_at_route_end(
    route: List[int],
    customer: int,
    depot: int,
) -> bool:
    """
    Vérifie si le client est à une extrémité de la route.

    Dans Clarke & Wright, on ne fusionne deux routes que si les clients
    concernés sont placés aux extrémités de leurs routes respectives.
    """
    if len(route) < 3:
        return False

    return route[1] == customer or route[-2] == customer


def _merge_routes(
    route_i: List[int],
    route_j: List[int],
    i: int,
    j: int,
    depot: int,
) -> List[int] | None:
    """
    Fusionne deux routes en reliant les clients i et j.

    Les quatre cas possibles sont traités :
    - i en fin de route_i et j en début de route_j ;
    - i en début de route_i et j en fin de route_j ;
    - i en début de route_i et j en début de route_j ;
    - i en fin de route_i et j en fin de route_j.
    """

    i_at_start = route_i[1] == i
    i_at_end = route_i[-2] == i

    j_at_start = route_j[1] == j
    j_at_end = route_j[-2] == j

    # Cas 1 : depot ... i + j ... depot
    if i_at_end and j_at_start:
        return route_i[:-1] + route_j[1:]

    # Cas 2 : depot ... j + i ... depot
    if i_at_start and j_at_end:
        return route_j[:-1] + route_i[1:]

    # Cas 3 : i est au début et j est au début.
    # On inverse route_i pour placer i à la fin.
    if i_at_start and j_at_start:
        reversed_i = list(reversed(route_i))
        return reversed_i[:-1] + route_j[1:]

    # Cas 4 : i est à la fin et j est à la fin.
    # On inverse route_j pour placer j au début.
    if i_at_end and j_at_end:
        reversed_j = list(reversed(route_j))
        return route_i[:-1] + reversed_j[1:]

    return None


def compute_savings(instance: CVRPInstance) -> List[Tuple[float, int, int]]:
    """
    Calcule les économies de Clarke & Wright.

    Formule :
    saving(i, j) = c(depot, i) + c(depot, j) - c(i, j)

    Plus le saving est élevé, plus il est avantageux de mettre i et j
    dans une même route.
    """

    depot = instance.depot
    customers = instance.customers
    savings: List[Tuple[float, int, int]] = []

    for idx_i in range(len(customers)):
        for idx_j in range(idx_i + 1, len(customers)):
            i = customers[idx_i]
            j = customers[idx_j]

            saving = (
                instance.distance(depot, i)
                + instance.distance(depot, j)
                - instance.distance(i, j)
            )

            savings.append((saving, i, j))

    savings.sort(reverse=True, key=lambda item: item[0])
    return savings


def clarke_wright_savings(
    instance: CVRPInstance,
    allow_negative_savings: bool = False,
) -> List[List[int]]:
    """
    Construit une solution CVRP avec l'algorithme de Clarke & Wright Savings.

    Étapes :
    1. Créer une route individuelle pour chaque client :
       depot -> client -> depot.
    2. Calculer les économies de fusion entre chaque paire de clients.
    3. Trier les économies par ordre décroissant.
    4. Fusionner les routes lorsque :
       - les clients sont aux extrémités ;
       - ils appartiennent à deux routes différentes ;
       - la capacité du véhicule n'est pas dépassée.

    Cette heuristique est plus forte que Nearest Neighbor et constitue une
    baseline classique importante pour le CVRP.
    """

    depot = instance.depot

    routes: List[List[int]] = [
        [depot, customer, depot]
        for customer in instance.customers
    ]

    savings = compute_savings(instance)

    for saving, i, j in savings:
        if saving < 0 and not allow_negative_savings:
            continue

        route_i_idx = _find_route_containing_customer(routes, i)
        route_j_idx = _find_route_containing_customer(routes, j)

        if route_i_idx is None or route_j_idx is None:
            continue

        if route_i_idx == route_j_idx:
            continue

        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]

        if not _is_customer_at_route_end(route_i, i, depot):
            continue

        if not _is_customer_at_route_end(route_j, j, depot):
            continue

        combined_load = (
            _route_load(instance, route_i)
            + _route_load(instance, route_j)
        )

        if combined_load > instance.capacity:
            continue

        merged_route = _merge_routes(
            route_i=route_i,
            route_j=route_j,
            i=i,
            j=j,
            depot=depot,
        )

        if merged_route is None:
            continue

        # On supprime les deux anciennes routes.
        # On supprime d'abord l'indice le plus grand pour ne pas décaler l'autre.
        for idx in sorted([route_i_idx, route_j_idx], reverse=True):
            routes.pop(idx)

        routes.append(merged_route)

    return normalize_solution(routes, depot)