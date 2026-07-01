from collections import Counter
from dataclasses import dataclass
from typing import List

from src.cvrp.core.cost import compute_gap, solution_cost
from src.cvrp.data.instance import CVRPInstance


@dataclass
class ValidationResult:
    is_valid: bool
    cost: float | None
    gap_percent: float | None
    route_loads: List[int]
    errors: List[str]
    warnings: List[str]
    missing_customers: List[int]
    duplicated_customers: List[int]
    unknown_nodes: List[int]


def normalize_solution(routes: List[List[int]], depot: int) -> List[List[int]]:
    """
    Normalise des routes pour qu'elles commencent et se terminent au dépôt.

    Cette fonction est utile pour les heuristiques, mais la validation ne doit pas
    corriger silencieusement une solution invalide. La validation vérifie donc les
    routes telles qu'elles sont fournies.
    """
    normalized: List[List[int]] = []

    for route in routes:
        if not route:
            continue

        fixed_route = list(route)

        if fixed_route[0] != depot:
            fixed_route = [depot, *fixed_route]

        if fixed_route[-1] != depot:
            fixed_route = [*fixed_route, depot]

        normalized.append(fixed_route)

    return normalized


def validate_solution(
    instance: CVRPInstance,
    routes: List[List[int]],
    require_all_customers: bool = True,
    check_vehicle_count: bool = True,
) -> ValidationResult:
    """
    Vérifie qu'une solution CVRP respecte les contraintes de base.

    Contraintes vérifiées :
    - chaque route commence au dépôt ;
    - chaque route se termine au dépôt ;
    - aucun dépôt ne se trouve au milieu d'une route ;
    - chaque client est visité exactement une fois ;
    - aucune route ne dépasse la capacité du véhicule ;
    - le nombre de routes ne dépasse pas le nombre de véhicules disponibles ;
    - aucun noeud inconnu n'apparaît dans la solution.
    """
    errors: List[str] = []
    warnings: List[str] = []
    route_loads: List[int] = []
    visited_customers: List[int] = []
    unknown_nodes_set = set()

    if routes is None:
        routes = []

    if not routes:
        errors.append("La solution ne contient aucune route.")

    # Correction importante :
    # dans un CVRP strict, utiliser plus de routes que de véhicules disponibles
    # rend la solution invalide.
    if check_vehicle_count and instance.vehicle_count is not None:
        if len(routes) > instance.vehicle_count:
            errors.append(
                f"Nombre de routes supérieur au nombre de véhicules : "
                f"{len(routes)} > {instance.vehicle_count}."
            )

    valid_nodes = set(instance.nodes)

    for route_idx, route in enumerate(routes, start=1):
        if not route:
            errors.append(f"Route {route_idx} est vide.")
            route_loads.append(0)
            continue

        if route[0] != instance.depot:
            errors.append(
                f"Route {route_idx} ne commence pas au dépôt {instance.depot}."
            )

        if route[-1] != instance.depot:
            errors.append(
                f"Route {route_idx} ne se termine pas au dépôt {instance.depot}."
            )

        # Vérification des noeuds inconnus
        for node in route:
            if node not in valid_nodes:
                unknown_nodes_set.add(node)

        # Les clients doivent être entre les deux dépôts.
        # Exemple valide : [depot, client_1, client_2, depot]
        internal_nodes = route[1:-1]

        if instance.depot in internal_nodes:
            errors.append(
                f"Route {route_idx} invalide : le dépôt apparaît au milieu de la route."
            )

        route_customers = [
            node
            for node in internal_nodes
            if node != instance.depot and node in valid_nodes
        ]

        visited_customers.extend(route_customers)

        load = sum(instance.demands.get(node, 0) for node in route_customers)
        route_loads.append(load)

        if load > instance.capacity:
            errors.append(
                f"Route {route_idx} dépasse la capacité : "
                f"{load} > {instance.capacity}."
            )

    duplicated_customers: List[int] = []
    missing_customers: List[int] = []

    customer_counts = Counter(visited_customers)

    duplicated_customers = sorted(
        customer
        for customer, count in customer_counts.items()
        if count > 1
    )

    if require_all_customers:
        missing_customers = sorted(
            customer
            for customer in instance.customers
            if customer_counts[customer] == 0
        )

    for customer in duplicated_customers:
        errors.append(
            f"Client {customer} visité {customer_counts[customer]} fois."
        )

    for customer in missing_customers:
        errors.append(f"Client {customer} non visité.")

    unknown_nodes = sorted(unknown_nodes_set)

    for node in unknown_nodes:
        errors.append(f"Noeud inconnu dans la solution : {node}.")

    cost = None
    gap_percent = None

    # On calcule le coût uniquement si aucun noeud inconnu n'est présent.
    # Sinon solution_cost risque d'échouer.
    if routes and not unknown_nodes:
        try:
            cost = solution_cost(instance, routes)
            gap_percent = (
                compute_gap(cost, instance.reference_cost)
                if cost is not None
                else None
            )
        except Exception as exc:
            errors.append(f"Erreur lors du calcul du coût : {exc}")
            cost = None
            gap_percent = None

    return ValidationResult(
        is_valid=len(errors) == 0,
        cost=cost,
        gap_percent=gap_percent,
        route_loads=route_loads,
        errors=errors,
        warnings=warnings,
        missing_customers=missing_customers,
        duplicated_customers=duplicated_customers,
        unknown_nodes=unknown_nodes,
    )