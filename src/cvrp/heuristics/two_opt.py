from typing import List, Tuple

from src.cvrp.data.instance import CVRPInstance
from src.cvrp.core.validation import normalize_solution


def two_opt_route(
    instance: CVRPInstance,
    route: List[int],
    max_iterations: int = 1000,
) -> List[int]:
    """
    Applique 2-opt sur une seule route.

    2-opt améliore l'ordre de visite des clients dans une route en inversant
    des segments lorsque cela réduit la distance totale.

    Important :
    - La route doit commencer et finir par le dépôt.
    - Le dépôt n'est jamais déplacé.
    - La capacité n'est pas modifiée, car les mêmes clients restent dans la route.
    """

    if len(route) <= 4:
        return route[:]

    best_route = route[:]
    best_cost = instance.route_cost(best_route)

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # On évite les positions 0 et len(route)-1, car ce sont les dépôts.
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue

                candidate = (
                    best_route[:i]
                    + list(reversed(best_route[i:j]))
                    + best_route[j:]
                )

                candidate_cost = instance.route_cost(candidate)

                if candidate_cost < best_cost:
                    best_route = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

    return best_route


def two_opt_solution(
    instance: CVRPInstance,
    routes: List[List[int]],
    max_iterations_per_route: int = 1000,
) -> List[List[int]]:
    """
    Applique 2-opt à toutes les routes d'une solution CVRP.

    Cette fonction ne change pas l'affectation des clients aux véhicules.
    Elle améliore seulement l'ordre de visite à l'intérieur de chaque route.
    """

    depot = instance.depot
    normalized_routes = normalize_solution(routes, depot)

    improved_routes = [
        two_opt_route(
            instance=instance,
            route=route,
            max_iterations=max_iterations_per_route,
        )
        for route in normalized_routes
    ]

    return normalize_solution(improved_routes, depot)


def two_opt_gain(
    instance: CVRPInstance,
    before_routes: List[List[int]],
    after_routes: List[List[int]],
) -> Tuple[float, float, float]:
    """
    Calcule le gain obtenu par 2-opt.

    Retourne :
    - coût avant ;
    - coût après ;
    - gain absolu.
    """

    before_cost = instance.solution_cost(before_routes)
    after_cost = instance.solution_cost(after_routes)
    gain = before_cost - after_cost

    return before_cost, after_cost, gain