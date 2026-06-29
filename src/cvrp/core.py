from __future__ import annotations

import numpy as np

from cvrp.data import CVRPInstance


Route = list[int]
Solution = list[Route]


def route_demand(route: Route, instance: CVRPInstance) -> int:
    """
    Calcule la demande totale d'une route.
    Le dépôt a normalement une demande de 0.
    """
    return int(sum(instance.demands[node] for node in route if node != instance.depot))


def route_cost(route: Route, instance: CVRPInstance) -> float:
    """
    Calcule le coût total d'une route.
    
    Exemple :
    route = [0, 5, 12, 7, 0]
    coût = d(0,5) + d(5,12) + d(12,7) + d(7,0)
    """
    total = 0.0

    for i in range(len(route) - 1):
        current_node = route[i]
        next_node = route[i + 1]
        total += instance.distance_matrix[current_node][next_node]

    return float(round(total, 2))


def solution_cost(solution: Solution, instance: CVRPInstance) -> float:
    """
    Calcule le coût total d'une solution CVRP.
    """
    return float(round(sum(route_cost(route, instance) for route in solution), 2))


def is_route_feasible(route: Route, instance: CVRPInstance) -> bool:
    """
    Une route est faisable si :
    - elle commence au dépôt ;
    - elle finit au dépôt ;
    - elle respecte la capacité du véhicule.
    """
    if len(route) < 2:
        return False

    starts_at_depot = route[0] == instance.depot
    ends_at_depot = route[-1] == instance.depot
    demand_ok = route_demand(route, instance) <= instance.capacity

    return starts_at_depot and ends_at_depot and demand_ok


def validate_solution(solution: Solution, instance: CVRPInstance) -> tuple[bool, list[str]]:
    """
    Vérifie si une solution CVRP est valide.
    
    Une solution est valide si :
    - toutes les routes commencent et finissent au dépôt ;
    - toutes les routes respectent la capacité ;
    - chaque client est visité exactement une fois ;
    - aucun client n'est oublié ;
    - aucun client n'est répété.
    """
    errors: list[str] = []

    all_nodes = set(range(len(instance.demands)))
    clients = all_nodes - {instance.depot}

    visited_clients: list[int] = []

    for route_index, route in enumerate(solution, start=1):
        if not is_route_feasible(route, instance):
            errors.append(f"Route {route_index} non faisable : {route}")

        for node in route:
            if node != instance.depot:
                visited_clients.append(node)

    visited_set = set(visited_clients)

    missing_clients = clients - visited_set
    duplicated_clients = {
        node for node in visited_clients if visited_clients.count(node) > 1
    }

    if missing_clients:
        errors.append(f"Clients manquants : {sorted(missing_clients)}")

    if duplicated_clients:
        errors.append(f"Clients dupliqués : {sorted(duplicated_clients)}")

    return len(errors) == 0, errors


def compute_gap(cost: float, reference_cost: float) -> float:
    """
    Calcule l'écart en pourcentage par rapport à une valeur de référence.
    """
    return round(((cost - reference_cost) / reference_cost) * 100, 2)