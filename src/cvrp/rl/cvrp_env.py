from __future__ import annotations

import random
from typing import Any

import numpy as np

from cvrp.data import CVRPInstance
from cvrp.core import Solution, validate_solution, solution_cost


class CVRPEnv:
    """
    Environnement RL minimal pour le CVRP.

    L'agent construit une solution en choisissant successivement :
    - un client non visité ;
    - ou le dépôt pour fermer la route courante.

    Convention :
    - action = id du nœud ;
    - action = depot signifie retour au dépôt.
    """

    def __init__(
        self,
        instance: CVRPInstance,
        max_routes: int | None = None,
        invalid_action_penalty: float = -1000.0,
        incomplete_solution_penalty: float = -5000.0,
        completion_bonus: float = 100.0,
        auto_return_to_depot_when_done: bool = True,
    ):
        self.instance = instance
        self.depot = instance.depot
        self.max_routes = max_routes

        self.invalid_action_penalty = invalid_action_penalty
        self.incomplete_solution_penalty = incomplete_solution_penalty
        self.completion_bonus = completion_bonus
        self.auto_return_to_depot_when_done = auto_return_to_depot_when_done

        self.n_nodes = len(instance.demands)
        self.clients = [node for node in range(self.n_nodes) if node != self.depot]

        self.current_node: int = self.depot
        self.remaining_capacity: int = instance.capacity
        self.visited: np.ndarray = np.zeros(self.n_nodes, dtype=bool)

        self.routes: Solution = []
        self.current_route: list[int] = [self.depot]

        self.total_distance: float = 0.0
        self.done: bool = False
        self.invalid_actions_count: int = 0
        self.steps_count: int = 0

        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """
        Réinitialise l'environnement au début d'un épisode.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_node = self.depot
        self.remaining_capacity = self.instance.capacity

        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.visited[self.depot] = True

        self.routes = []
        self.current_route = [self.depot]

        self.total_distance = 0.0
        self.done = False
        self.invalid_actions_count = 0
        self.steps_count = 0

        return self._get_observation()

    def _all_clients_visited(self) -> bool:
        """
        Vérifie si tous les clients ont été servis.
        """
        return all(self.visited[client] for client in self.clients)

    def _distance(self, from_node: int, to_node: int) -> float:
        """
        Retourne la distance entre deux nœuds.
        """
        return float(self.instance.distance_matrix[from_node][to_node])

    def _close_current_route(self) -> float:
        """
        Ferme la route courante en retournant au dépôt.
        Retourne la distance ajoutée.
        """
        if self.current_node == self.depot:
            return 0.0

        distance_to_depot = self._distance(self.current_node, self.depot)

        self.current_route.append(self.depot)
        self.routes.append(self.current_route)

        self.total_distance += distance_to_depot

        self.current_node = self.depot
        self.remaining_capacity = self.instance.capacity
        self.current_route = [self.depot]

        return distance_to_depot

    def get_valid_actions(self) -> list[int]:
        """
        Retourne les actions actuellement faisables.

        Actions possibles :
        - clients non visités dont la demande respecte la capacité restante ;
        - dépôt, si le véhicule est actuellement en tournée.
        """
        if self.done:
            return []

        valid_actions: list[int] = []

        for client in self.clients:
            if self.visited[client]:
                continue

            demand = int(self.instance.demands[client])

            if demand <= self.remaining_capacity:
                valid_actions.append(client)

        # Le retour au dépôt est autorisé seulement si on est sorti du dépôt.
        if self.current_node != self.depot:
            valid_actions.append(self.depot)

        return valid_actions

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Exécute une action.

        Retour :
        - observation ;
        - reward ;
        - done ;
        - info.
        """
        if self.done:
            raise RuntimeError("L'épisode est terminé. Appelle reset() pour recommencer.")

        self.steps_count += 1

        info: dict[str, Any] = {
            "action": action,
            "valid_action": True,
            "message": "",
        }

        # Cas 1 : action invalide hors bornes.
        if action < 0 or action >= self.n_nodes:
            return self._invalid_action(
                message=f"Action hors limites : {action}",
                info=info,
            )

        # Cas 2 : retour au dépôt.
        if action == self.depot:
            return self._step_return_to_depot(info)

        # Cas 3 : visite d'un client.
        return self._step_visit_client(action, info)
















    def _step_return_to_depot(
        self,
        info: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Action : retour au dépôt.
        """
        if self.current_node == self.depot:
            return self._invalid_action(
                message="Retour inutile : le véhicule est déjà au dépôt.",
                info=info,
            )

        distance_added = self._close_current_route()
        reward = -distance_added

        info["message"] = "Retour au dépôt."
        info["distance_added"] = distance_added

        if self._all_clients_visited():
            self.done = True
            reward += self.completion_bonus
            info["message"] = "Solution complète terminée."

        elif self.max_routes is not None and len(self.routes) >= self.max_routes:
            self.done = True
            missing = self._missing_clients()
            penalty = self.incomplete_solution_penalty * len(missing)
            reward += penalty
            info["valid_action"] = False
            info["message"] = (
                f"Nombre maximal de routes atteint. Clients non servis : {missing}"
            )

        return self._get_observation(), reward, self.done, info

    # def _step_return_to_depot(
    #     self,
    #     info: dict[str, Any],
    # ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    #     """
    #     Action : retour au dépôt.

    #     Version renforcée :
    #     - pénalise les retours trop précoces ;
    #     - encourage le bon remplissage du véhicule ;
    #     - garde la récompense principale basée sur la distance.
    #     """
    #     if self.current_node == self.depot:
    #         return self._invalid_action(
    #             message="Retour inutile : le véhicule est déjà au dépôt.",
    #             info=info,
    #         )

    #     used_capacity = self.instance.capacity - self.remaining_capacity
    #     used_ratio = used_capacity / self.instance.capacity

    #     distance_added = self._close_current_route()
    #     reward = -distance_added

    #     # Pénalité si le véhicule revient trop vide alors qu'il reste des clients.
    #     if not self._all_clients_visited():
    #         underload_ratio = 1 - used_ratio
    #         reward -= 50 * underload_ratio

    #     # Petite récompense si la route est bien remplie.
    #     if used_ratio >= 0.75:
    #         reward += 20

    #     # Pénalité si on dépasse le nombre de véhicules indiqué.
    #     if self.instance.vehicle_count is not None:
    #         if len(self.routes) > self.instance.vehicle_count:
    #             reward -= 200 * (len(self.routes) - self.instance.vehicle_count)

    #     info["message"] = "Retour au dépôt."
    #     info["distance_added"] = distance_added
    #     info["used_capacity"] = used_capacity
    #     info["used_ratio"] = round(used_ratio, 2)

    #     if self._all_clients_visited():
    #         self.done = True
    #         reward += self.completion_bonus
    #         info["message"] = "Solution complète terminée."

    #     elif self.max_routes is not None and len(self.routes) >= self.max_routes:
    #         self.done = True
    #         missing = self._missing_clients()
    #         penalty = self.incomplete_solution_penalty * len(missing)
    #         reward += penalty
    #         info["valid_action"] = False
    #         info["message"] = (
    #             f"Nombre maximal de routes atteint. Clients non servis : {missing}"
    #         )

    #     return self._get_observation(), reward, self.done, info



    def _step_visit_client(
        self,
        client: int,
        info: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Action : visiter un client.
        """
        if self.visited[client]:
            return self._invalid_action(
                message=f"Client déjà visité : {client}",
                info=info,
            )

        demand = int(self.instance.demands[client])

        if demand > self.remaining_capacity:
            return self._invalid_action(
                message=(
                    f"Capacité insuffisante pour le client {client}. "
                    f"Demande={demand}, capacité restante={self.remaining_capacity}"
                ),
                info=info,
            )

        distance_added = self._distance(self.current_node, client)

        self.total_distance += distance_added
        self.remaining_capacity -= demand
        self.current_node = client
        self.current_route.append(client)
        self.visited[client] = True

        reward = -distance_added

        info["message"] = f"Client {client} visité."
        info["distance_added"] = distance_added
        info["remaining_capacity"] = self.remaining_capacity

        # Si tous les clients sont visités, on ferme automatiquement la dernière route.
        if self._all_clients_visited() and self.auto_return_to_depot_when_done:
            distance_to_depot = self._close_current_route()
            reward -= distance_to_depot
            reward += self.completion_bonus
            self.done = True

            info["message"] = "Tous les clients sont servis. Retour automatique au dépôt."
            info["distance_to_depot"] = distance_to_depot

        return self._get_observation(), reward, self.done, info

    def _invalid_action(
        self,
        message: str,
        info: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Gère une action invalide.
        """
        self.invalid_actions_count += 1

        info["valid_action"] = False
        info["message"] = message

        # Sécurité pour éviter les épisodes infinis.
        if self.invalid_actions_count >= self.n_nodes * 2:
            self.done = True
            info["message"] += " Trop d'actions invalides, épisode arrêté."

        return self._get_observation(), self.invalid_action_penalty, self.done, info

    def _missing_clients(self) -> list[int]:
        """
        Retourne la liste des clients non encore servis.
        """
        return [client for client in self.clients if not self.visited[client]]

    def _get_observation(self) -> dict[str, Any]:
        """
        Retourne l'état actuel de l'environnement.
        """
        return {
            "current_node": self.current_node,
            "remaining_capacity": self.remaining_capacity,
            "visited_mask": self.visited.astype(int).tolist(),
            "missing_clients": self._missing_clients(),
            "routes_count": len(self.routes),
            "current_route": self.current_route.copy(),
            "total_distance": round(self.total_distance, 2),
            "done": self.done,
        }

    def get_observation_vector(self) -> np.ndarray:
        """
        Retourne une version vectorielle simplifiée de l'état.

        Cette version pourra servir plus tard pour un agent neuronal.
        """
        coords = self.instance.coords
        max_x = max(coords[:, 0]) if max(coords[:, 0]) != 0 else 1
        max_y = max(coords[:, 1]) if max(coords[:, 1]) != 0 else 1

        current_x = coords[self.current_node][0] / max_x
        current_y = coords[self.current_node][1] / max_y

        remaining_capacity_ratio = self.remaining_capacity / self.instance.capacity

        visited_mask = self.visited.astype(float)

        return np.concatenate(
            [
                np.array(
                    [
                        current_x,
                        current_y,
                        remaining_capacity_ratio,
                    ],
                    dtype=float,
                ),
                visited_mask,
            ]
        )

    def get_solution(self) -> Solution:
        """
        Retourne la solution construite.

        Si une route est en cours et contient déjà des clients,
        elle est ajoutée sous forme fermée pour lecture.
        """
        solution = [route.copy() for route in self.routes]

        if len(self.current_route) > 1:
            if self.current_route[-1] != self.depot:
                solution.append(self.current_route + [self.depot])
            else:
                solution.append(self.current_route.copy())

        return solution

    def summary(self) -> dict[str, Any]:
        """
        Retourne un résumé de l'épisode.
        """
        solution = self.get_solution()
        valid, errors = validate_solution(solution, self.instance)

        return {
            "solution": solution,
            "valid": valid,
            "errors": errors,
            "cost": solution_cost(solution, self.instance) if solution else 0.0,
            "routes": len(solution),
            "total_distance": round(self.total_distance, 2),
            "steps": self.steps_count,
            "invalid_actions": self.invalid_actions_count,
            "done": self.done,
        }