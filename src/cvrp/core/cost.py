from typing import List

from src.cvrp.data.instance import CVRPInstance


def route_cost(instance: CVRPInstance, route: List[int]) -> float:
    """
    Calcule le coût d'une route.

    Exemple :
    [1, 5, 8, 1]
    signifie : dépôt -> client 5 -> client 8 -> dépôt
    """
    return instance.route_cost(route)


def solution_cost(instance: CVRPInstance, routes: List[List[int]]) -> float:
    """
    Calcule le coût total d'une solution CVRP.
    """
    return instance.solution_cost(routes)


def route_demand(instance: CVRPInstance, route: List[int]) -> int:
    """
    Calcule la demande totale servie par une route.
    """
    return instance.route_demand(route)


def compute_gap(cost: float, reference_cost: float | None) -> float | None:
    """
    Calcule le gap en pourcentage par rapport à une valeur de référence.

    gap = ((coût_obtenu - coût_référence) / coût_référence) * 100
    """
    if reference_cost is None:
        return None

    if reference_cost == 0:
        return None

    return ((cost - reference_cost) / reference_cost) * 100


def summarize_solution(instance: CVRPInstance, routes: List[List[int]]) -> dict:
    """
    Produit un résumé simple d'une solution.

    Ce résumé sera utilisé plus tard pour générer les CSV de résultats.
    """
    cost = solution_cost(instance, routes)
    gap = compute_gap(cost, instance.reference_cost)

    return {
        "instance": instance.name,
        "cost": cost,
        "reference_cost": instance.reference_cost,
        "gap_percent": gap,
        "routes": len(routes),
        "vehicle_count": instance.vehicle_count,
        "capacity": instance.capacity,
    }