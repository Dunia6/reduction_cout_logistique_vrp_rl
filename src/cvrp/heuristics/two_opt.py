from __future__ import annotations

from cvrp.data import CVRPInstance
from cvrp.core import Route, Solution, route_cost


def two_opt_route(route: Route, instance: CVRPInstance) -> Route:
    """
    Applique 2-opt sur une seule route CVRP.

    La route doit commencer et finir par le dépôt.
    Exemple :
        [0, 5, 12, 7, 0]

    2-opt ne change pas les clients affectés à la route.
    Il change seulement leur ordre de visite.
    """
    if len(route) <= 4:
        return route.copy()

    best_route = route.copy()
    best_cost = route_cost(best_route, instance)

    improved = True

    while improved:
        improved = False

        # On évite i = 0 et j = dernier index pour ne pas inverser le dépôt.
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                candidate_route = (
                    best_route[:i]
                    + best_route[i:j + 1][::-1]
                    + best_route[j + 1:]
                )

                candidate_cost = route_cost(candidate_route, instance)

                if candidate_cost < best_cost:
                    best_route = candidate_route
                    best_cost = candidate_cost
                    improved = True

                    # On recommence la recherche à partir de la nouvelle route.
                    break

            if improved:
                break

    return best_route


def two_opt_solution(solution: Solution, instance: CVRPInstance) -> Solution:
    """
    Applique 2-opt sur toutes les routes d'une solution CVRP.
    """
    improved_solution: Solution = []

    for route in solution:
        improved_route = two_opt_route(route, instance)
        improved_solution.append(improved_route)

    return improved_solution