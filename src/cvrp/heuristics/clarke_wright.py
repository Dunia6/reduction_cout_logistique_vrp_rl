from __future__ import annotations

from cvrp.data import CVRPInstance
from cvrp.core import Route, Solution, route_demand


def compute_savings(instance: CVRPInstance) -> list[tuple[float, int, int]]:
    """
    Calcule les savings de Clarke & Wright pour toutes les paires de clients.

    saving(i, j) = c(0, i) + c(0, j) - c(i, j)

    Plus le saving est grand, plus la fusion de i et j est intéressante.
    """
    depot = instance.depot
    distance_matrix = instance.distance_matrix

    clients = [node for node in range(len(instance.demands)) if node != depot]

    savings: list[tuple[float, int, int]] = []

    for index_i in range(len(clients)):
        for index_j in range(index_i + 1, len(clients)):
            i = clients[index_i]
            j = clients[index_j]

            saving = (
                distance_matrix[depot][i]
                + distance_matrix[depot][j]
                - distance_matrix[i][j]
            )

            savings.append((float(saving), i, j))

    savings.sort(reverse=True, key=lambda item: item[0])

    return savings


def find_route_containing_client(routes: Solution, client: int) -> int | None:
    """
    Retourne l'index de la route contenant un client donné.
    """
    for route_index, route in enumerate(routes):
        if client in route[1:-1]:
            return route_index

    return None


def try_merge_routes(
    route_a: Route,
    route_b: Route,
    i: int,
    j: int,
    instance: CVRPInstance,
) -> Route | None:
    """
    Essaie de fusionner deux routes si i et j sont aux extrémités.

    Dans Clarke & Wright, on ne fusionne que si les clients concernés sont
    aux extrémités de leurs routes respectives.

    Exemple acceptable :
        route_a = [0, ..., i, 0]
        route_b = [0, j, ..., 0]
        fusion  = [0, ..., i, j, ..., 0]
    """
    depot = instance.depot

    inner_a = route_a[1:-1]
    inner_b = route_b[1:-1]

    if not inner_a or not inner_b:
        return None

    # Cas 1 : route_a finit par i, route_b commence par j
    if inner_a[-1] == i and inner_b[0] == j:
        merged_inner = inner_a + inner_b

    # Cas 2 : route_a commence par i, route_b finit par j
    elif inner_a[0] == i and inner_b[-1] == j:
        merged_inner = inner_b + inner_a

    # Cas 3 : les deux routes commencent par i et j
    # On inverse route_a pour que i devienne une extrémité de fin.
    elif inner_a[0] == i and inner_b[0] == j:
        merged_inner = inner_a[::-1] + inner_b

    # Cas 4 : les deux routes finissent par i et j
    # On inverse route_b pour que j devienne une extrémité de début.
    elif inner_a[-1] == i and inner_b[-1] == j:
        merged_inner = inner_a + inner_b[::-1]

    else:
        return None

    merged_route = [depot] + merged_inner + [depot]

    if route_demand(merged_route, instance) > instance.capacity:
        return None

    return merged_route


def clarke_wright_savings(
    instance: CVRPInstance,
    allow_negative_savings: bool = False,
) -> Solution:
    """
    Implémente l'heuristique Clarke & Wright Savings en version parallèle.

    Étapes :
    1. Créer une route par client : [depot, client, depot]
    2. Calculer les économies pour toutes les paires de clients
    3. Trier les économies par ordre décroissant
    4. Fusionner les routes lorsque :
       - les clients sont dans des routes différentes ;
       - les clients sont aux extrémités des routes ;
       - la capacité n'est pas dépassée.
    """
    depot = instance.depot
    clients = [node for node in range(len(instance.demands)) if node != depot]

    routes: Solution = [[depot, client, depot] for client in clients]

    savings = compute_savings(instance)

    for saving, i, j in savings:
        if saving <= 0 and not allow_negative_savings:
            continue

        route_i_index = find_route_containing_client(routes, i)
        route_j_index = find_route_containing_client(routes, j)

        if route_i_index is None or route_j_index is None:
            continue

        if route_i_index == route_j_index:
            continue

        route_i = routes[route_i_index]
        route_j = routes[route_j_index]

        merged_route = try_merge_routes(
            route_a=route_i,
            route_b=route_j,
            i=i,
            j=j,
            instance=instance,
        )

        if merged_route is None:
            continue

        # Remplacer les deux anciennes routes par la nouvelle route fusionnée.
        new_routes: Solution = []

        for route_index, route in enumerate(routes):
            if route_index not in {route_i_index, route_j_index}:
                new_routes.append(route)

        new_routes.append(merged_route)
        routes = new_routes

    return routes