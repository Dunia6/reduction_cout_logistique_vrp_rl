from typing import List, Set

from src.cvrp.data.instance import CVRPInstance
from src.cvrp.core.validation import normalize_solution


def nearest_neighbor_cvrp(instance: CVRPInstance) -> List[List[int]]:
    """
    Construit une solution CVRP avec l'heuristique du plus proche voisin.

    Principe :
    - On part du dépôt.
    - On choisit le client non visité le plus proche qui respecte la capacité restante.
    - Si aucun client ne peut être ajouté, on ferme la route et on revient au dépôt.
    - On répète jusqu'à visiter tous les clients.

    Cette méthode est une baseline simple. Elle est utile pour comparer les méthodes
    plus avancées, mais elle n'est pas censée produire les meilleures routes.
    """

    depot = instance.depot
    unvisited: Set[int] = set(instance.customers)
    routes: List[List[int]] = []

    while unvisited:
        current_node = depot
        remaining_capacity = instance.capacity
        route = [depot]

        while True:
            feasible_customers = [
                customer
                for customer in unvisited
                if instance.demands[customer] <= remaining_capacity
            ]

            if not feasible_customers:
                break

            next_customer = min(
                feasible_customers,
                key=lambda customer: instance.distance(current_node, customer),
            )

            route.append(next_customer)
            unvisited.remove(next_customer)

            remaining_capacity -= instance.demands[next_customer]
            current_node = next_customer

        route.append(depot)
        routes.append(route)

    return normalize_solution(routes, depot)