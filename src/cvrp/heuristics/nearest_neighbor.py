from __future__ import annotations

from cvrp.data import CVRPInstance
from cvrp.core import Solution, Route


def nearest_neighbor_cvrp(
    instance: CVRPInstance,
    max_vehicles: int | None = None,
) -> Solution:
    """
    Construit une solution CVRP avec une heuristique Nearest Neighbor.

    Principe :
    - on commence au dépôt ;
    - on choisit le client non visité le plus proche ;
    - on vérifie que sa demande respecte la capacité restante ;
    - si aucun client faisable ne peut être ajouté, on retourne au dépôt ;
    - on démarre une nouvelle route jusqu'à servir tous les clients.

    max_vehicles :
    - si None : nombre de véhicules non limité ;
    - si un entier est fourni : l'algorithme ne doit pas dépasser ce nombre.
    """
    depot = instance.depot
    capacity = instance.capacity
    distance_matrix = instance.distance_matrix
    demands = instance.demands

    unvisited_clients = set(range(len(demands)))
    unvisited_clients.remove(depot)

    solution: Solution = []

    while unvisited_clients:
        if max_vehicles is not None and len(solution) >= max_vehicles:
            raise ValueError(
                f"Impossible de servir tous les clients avec {max_vehicles} véhicules "
                "en utilisant cette heuristique gloutonne."
            )

        route: Route = [depot]
        current_node = depot
        remaining_capacity = capacity

        while True:
            feasible_clients = [
                client
                for client in unvisited_clients
                if demands[client] <= remaining_capacity
            ]

            if not feasible_clients:
                break

            next_client = min(
                feasible_clients,
                key=lambda client: distance_matrix[current_node][client],
            )

            route.append(next_client)
            unvisited_clients.remove(next_client)

            remaining_capacity -= demands[next_client]
            current_node = next_client

        route.append(depot)
        solution.append(route)

    return solution