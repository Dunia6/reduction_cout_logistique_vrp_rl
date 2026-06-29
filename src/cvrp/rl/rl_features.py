from __future__ import annotations

import numpy as np
import torch

from cvrp.rl.cvrp_env import CVRPEnv


def build_node_features(env: CVRPEnv) -> torch.Tensor:
    """
    Construit les features de chaque nœud.

    Pour chaque nœud, on encode :
    - x normalisé ;
    - y normalisé ;
    - demande normalisée ;
    - est-ce le dépôt ;
    - est-ce le nœud courant ;
    - est-ce déjà visité ;
    - capacité restante ;
    - nombre de routes utilisées ;
    - fraction de clients restants.
    """
    instance = env.instance
    coords = instance.coords
    demands = instance.demands

    max_x = np.max(coords[:, 0]) or 1
    max_y = np.max(coords[:, 1]) or 1
    max_demand = np.max(demands) or 1

    n_nodes = len(demands)

    remaining_capacity_ratio = env.remaining_capacity / instance.capacity

    if instance.vehicle_count:
        routes_ratio = len(env.routes) / instance.vehicle_count
    else:
        routes_ratio = 0.0

    missing_count = len(env._missing_clients())
    clients_count = len(env.clients)
    missing_ratio = missing_count / clients_count if clients_count else 0.0

    features = []

    for node in range(n_nodes):
        x = coords[node][0] / max_x
        y = coords[node][1] / max_y

        demand = demands[node] / max_demand

        is_depot = 1.0 if node == env.depot else 0.0
        is_current = 1.0 if node == env.current_node else 0.0
        is_visited = 1.0 if env.visited[node] else 0.0

        features.append(
            [
                x,
                y,
                demand,
                is_depot,
                is_current,
                is_visited,
                remaining_capacity_ratio,
                routes_ratio,
                missing_ratio,
            ]
        )

    return torch.tensor(features, dtype=torch.float32)


def build_action_mask(env: CVRPEnv) -> torch.Tensor:
    """
    Construit un masque d'actions.

    True  = action autorisée
    False = action interdite

    Conditions respectées :
    - pas de client déjà visité ;
    - pas de dépassement de capacité ;
    - pas de retour inutile au dépôt ;
    - pas de dépassement du nombre de véhicules de référence.
    """
    n_nodes = env.n_nodes
    mask = torch.zeros(n_nodes, dtype=torch.bool)

    if env.done:
        return mask

    valid_clients = []

    for client in env.clients:
        if env.visited[client]:
            continue

        demand = int(env.instance.demands[client])

        if demand <= env.remaining_capacity:
            valid_clients.append(client)

    for client in valid_clients:
        mask[client] = True

    # Retour au dépôt.
    if env.current_node != env.depot:
        all_clients_visited = env._all_clients_visited()

        if all_clients_visited:
            mask[env.depot] = True
        else:
            if env.max_routes is None:
                mask[env.depot] = True
            else:
                # Si on ferme la route actuelle, le nombre de routes devient len(routes) + 1.
                # On autorise seulement si on peut encore ouvrir une autre route ensuite.
                closing_route_count = len(env.routes) + 1

                if closing_route_count < env.max_routes:
                    mask[env.depot] = True

    return mask