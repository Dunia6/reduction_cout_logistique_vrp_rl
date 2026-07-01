from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from src.cvrp.data.instance import CVRPInstance
from src.cvrp.core.validation import normalize_solution, validate_solution
from src.cvrp.rl.features import (
    build_node_features,
    build_q_learning_state,
    get_valid_actions,
    get_valid_action_mask,
)


@dataclass
class CVRPEnv:
    """
    Environnement CVRP commun pour les modèles RL.

    Il sera utilisé pour :
    - Q-learning ;
    - POMO-style Active Search ;
    - tests de masquage des actions ;
    - génération de métriques d'entraînement.

    Logique :
    - une action correspond au choix du prochain noeud ;
    - un client ne peut être choisi qu'une seule fois ;
    - un client ne peut être choisi que si sa demande respecte la capacité restante ;
    - le dépôt permet de fermer la route courante ;
    - la solution est terminée lorsque tous les clients sont visités et que le véhicule est revenu au dépôt.
    """

    instance: CVRPInstance
    max_routes: int | None = None
    invalid_action_penalty: float = -1000.0
    incomplete_solution_penalty: float = -5000.0
    completion_bonus: float = 100.0
    distance_reward_scale: float = 1.0

    current_node: int = field(init=False)
    remaining_capacity: int = field(init=False)
    visited_customers: Set[int] = field(init=False)
    current_route: List[int] = field(init=False)
    closed_routes: List[List[int]] = field(init=False)
    done: bool = field(init=False)
    total_distance: float = field(init=False)
    total_reward: float = field(init=False)
    steps: int = field(init=False)

    def __post_init__(self):
        if self.max_routes is None:
            self.max_routes = self.instance.vehicle_count

        self.reset()

    def reset(self) -> Dict[str, Any]:
        """
        Réinitialise l'environnement.
        """

        depot = self.instance.depot

        self.current_node = depot
        self.remaining_capacity = self.instance.capacity
        self.visited_customers = set()
        self.current_route = [depot]
        self.closed_routes = []
        self.done = False
        self.total_distance = 0.0
        self.total_reward = 0.0
        self.steps = 0

        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """
        Retourne l'observation actuelle.
        """

        return {
            "current_node": self.current_node,
            "remaining_capacity": self.remaining_capacity,
            "visited_customers": set(self.visited_customers),
            "current_route": list(self.current_route),
            "closed_routes": [list(route) for route in self.closed_routes],
            "done": self.done,
            "total_distance": self.total_distance,
            "steps": self.steps,
            "q_state": self.get_q_learning_state(),
            "valid_action_mask": self.get_valid_action_mask(),
        }

    def get_q_learning_state(self) -> tuple:
        """
        Retourne l'état discret utilisé par Q-learning.
        """

        return build_q_learning_state(
            instance=self.instance,
            current_node=self.current_node,
            remaining_capacity=self.remaining_capacity,
            visited_customers=self.visited_customers,
            closed_routes=self.closed_routes,
        )

    def get_node_features(self) -> List[List[float]]:
        """
        Retourne les features de noeuds pour les modèles neuronaux.
        """

        return build_node_features(
            instance=self.instance,
            current_node=self.current_node,
            remaining_capacity=self.remaining_capacity,
            visited_customers=self.visited_customers,
            current_route=self.current_route,
            closed_routes=self.closed_routes,
            max_routes=self.max_routes,
        )

    def get_valid_action_mask(self) -> Dict[int, bool]:
        """
        Retourne le masque d'actions valides.
        """

        return get_valid_action_mask(
            instance=self.instance,
            current_node=self.current_node,
            remaining_capacity=self.remaining_capacity,
            visited_customers=self.visited_customers,
            current_route=self.current_route,
            closed_routes=self.closed_routes,
            max_routes=self.max_routes,
        )

    def get_valid_actions(self) -> List[int]:
        """
        Retourne la liste des actions valides.
        """

        return get_valid_actions(
            instance=self.instance,
            current_node=self.current_node,
            remaining_capacity=self.remaining_capacity,
            visited_customers=self.visited_customers,
            current_route=self.current_route,
            closed_routes=self.closed_routes,
            max_routes=self.max_routes,
        )

    def step(self, action: int):
        """
        Exécute une action.

        Retour :
        - observation suivante ;
        - reward ;
        - done ;
        - info.
        """

        if self.done:
            return self.get_observation(), 0.0, True, {
                "message": "Episode déjà terminé."
            }

        self.steps += 1

        valid_actions = self.get_valid_actions()

        if action not in valid_actions:
            reward = self.invalid_action_penalty
            self.total_reward += reward

            info = {
                "valid_action": False,
                "reason": "Action invalide.",
                "action": action,
                "valid_actions": valid_actions,
            }

            return self.get_observation(), reward, self.done, info

        depot = self.instance.depot
        previous_node = self.current_node
        distance = self.instance.distance(previous_node, action)

        self.total_distance += distance

        # Reward principal : minimiser la distance.
        reward = -(distance / self.distance_reward_scale)

        # Cas 1 : retour au dépôt.
        if action == depot:
            self.current_route.append(depot)

            normalized_route = normalize_solution(
                [self.current_route],
                depot,
            )[0]

            self.closed_routes.append(normalized_route)

            self.current_node = depot
            self.remaining_capacity = self.instance.capacity
            self.current_route = [depot]

        # Cas 2 : visite d'un client.
        else:
            self.current_route.append(action)
            self.visited_customers.add(action)

            self.remaining_capacity -= self.instance.demands[action]
            self.current_node = action

        # Vérification de fin normale.
        if self._all_customers_visited() and self.current_node == depot:
            self.done = True
            reward += self.completion_bonus

        # Vérification d'échec : plus aucune action utile possible.
        if not self.done:
            next_valid_actions = self.get_valid_actions()

            if not next_valid_actions:
                self.done = True

                if not self._all_customers_visited():
                    reward += self.incomplete_solution_penalty

        self.total_reward += reward

        info = {
            "valid_action": True,
            "action": action,
            "distance": distance,
            "all_customers_visited": self._all_customers_visited(),
            "routes_count": len(self.closed_routes),
        }

        return self.get_observation(), reward, self.done, info

    def _all_customers_visited(self) -> bool:
        return len(self.visited_customers) == self.instance.customer_count

    def get_solution(self) -> List[List[int]]:
        """
        Retourne la solution actuelle.

        Si une route est encore ouverte, elle est fermée automatiquement
        pour l'évaluation.
        """

        depot = self.instance.depot
        routes = [list(route) for route in self.closed_routes]

        if len(self.current_route) > 1:
            route = list(self.current_route)

            if route[-1] != depot:
                route.append(depot)

            routes.append(route)

        return normalize_solution(routes, depot)

    def validate_current_solution(self):
        """
        Valide la solution actuelle.
        """

        return validate_solution(
            self.instance,
            self.get_solution(),
        )