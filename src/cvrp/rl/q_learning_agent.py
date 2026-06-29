from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from cvrp.rl.cvrp_env import CVRPEnv


class QLearningAgent:
    """
    Agent Q-learning tabulaire simplifié pour le CVRP.

    L'agent apprend une valeur Q(state, action), où :
    - state décrit la situation courante ;
    - action est le prochain nœud à visiter ou le dépôt ;
    - Q estime l'intérêt de choisir cette action dans cet état.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: dict[tuple, dict[int, float]] = defaultdict(dict)

    def get_state_key(self, observation: dict[str, Any]) -> tuple:
        """
        Transforme l'observation en clé utilisable dans la table Q.

        Pour garder le modèle simple, on encode :
        - le nœud courant ;
        - la capacité restante ;
        - le masque des clients déjà visités.
        """
        return (
            observation["current_node"],
            observation["remaining_capacity"],
            tuple(observation["visited_mask"]),
        )

    def get_q_value(self, state_key: tuple, action: int) -> float:
        """
        Retourne Q(state, action).
        Si la valeur n'existe pas encore, elle vaut 0.
        """
        return self.q_table[state_key].get(action, 0.0)

    def choose_action(self, env: CVRPEnv, observation: dict[str, Any]) -> int:
        """
        Choisit une action selon une politique epsilon-greedy.

        - Avec probabilité epsilon : exploration aléatoire.
        - Sinon : exploitation de la meilleure valeur Q connue.

        Si toutes les valeurs Q sont encore nulles, on utilise une règle
        de départ proche du Nearest Neighbor pour éviter un comportement
        totalement désordonné.
        """
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            return env.depot

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_key = self.get_state_key(observation)

        q_values = {
            action: self.get_q_value(state_key, action)
            for action in valid_actions
        }

        max_q = max(q_values.values())

        best_actions = [
            action for action, value in q_values.items()
            if value == max_q
        ]

        # Si toutes les valeurs sont encore à 0, on choisit le client faisable
        # le plus proche pour stabiliser l'apprentissage.
        if max_q == 0:
            client_actions = [
                action for action in valid_actions
                if action != env.depot
            ]

            if client_actions:
                current_node = env.current_node

                return min(
                    client_actions,
                    key=lambda client: env.instance.distance_matrix[current_node][client],
                )

        return random.choice(best_actions)

    def update(
        self,
        state_key: tuple,
        action: int,
        reward: float,
        next_state_key: tuple | None,
        next_valid_actions: list[int],
        done: bool,
    ) -> None:
        """
        Met à jour Q(state, action) avec la formule du Q-learning.
        """
        old_q = self.get_q_value(state_key, action)

        if done or next_state_key is None or not next_valid_actions:
            next_max_q = 0.0
        else:
            next_max_q = max(
                self.get_q_value(next_state_key, next_action)
                for next_action in next_valid_actions
            )

        target = reward + self.discount_factor * next_max_q

        new_q = old_q + self.learning_rate * (target - old_q)

        self.q_table[state_key][action] = new_q

    def decay_epsilon(self) -> None:
        """
        Réduit progressivement epsilon pour passer de l'exploration
        vers l'exploitation.
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )