import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


State = Tuple[int, int, int, int]
Action = int


class QLearningAgent:
    """
    Agent Q-learning tabulaire pour le CVRP.

    État utilisé :
    - noeud courant ;
    - capacité restante ;
    - masque entier des clients visités ;
    - nombre de routes déjà fermées.

    Action :
    - choisir le prochain noeud à visiter ;
    - le dépôt est aussi une action lorsqu'un retour au dépôt est valide.

    Cet agent sert de baseline RL simple. Il n'est pas censé être aussi performant
    qu'un modèle neuronal moderne comme POMO, mais il permet de comparer une
    approche RL basique à des heuristiques classiques.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # q_table[state][action] = valeur Q
        self.q_table: Dict[State, Dict[Action, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def select_action(
        self,
        state: State,
        valid_actions: List[Action],
        greedy: bool = False,
    ) -> Action:
        """
        Sélectionne une action avec une stratégie epsilon-greedy.

        Si greedy=True, on choisit toujours la meilleure action connue.
        """

        if not valid_actions:
            raise ValueError("Aucune action valide disponible.")

        # Exploration.
        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitation.
        q_values = self.q_table[state]

        best_value = max(q_values[action] for action in valid_actions)
        best_actions = [
            action
            for action in valid_actions
            if q_values[action] == best_value
        ]

        return random.choice(best_actions)

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_valid_actions: List[Action],
        done: bool,
    ) -> None:
        """
        Met à jour la table Q selon la règle :

        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        """

        current_q = self.q_table[state][action]

        if done or not next_valid_actions:
            target = reward
        else:
            next_max_q = max(
                self.q_table[next_state][next_action]
                for next_action in next_valid_actions
            )
            target = reward + self.gamma * next_max_q

        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self) -> None:
        """
        Réduit epsilon après un épisode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def reset_epsilon(self) -> None:
        """
        Réinitialise epsilon à sa valeur initiale.
        """

        self.epsilon = self.epsilon_start

    def q_table_size(self) -> int:
        """
        Retourne le nombre d'états connus.
        """

        return len(self.q_table)

    def to_serializable(self) -> dict:
        """
        Convertit la Q-table en dictionnaire sérialisable JSON.
        """

        serialized_q_table = {}

        for state, action_values in self.q_table.items():
            state_key = "|".join(map(str, state))
            serialized_q_table[state_key] = {
                str(action): value
                for action, value in action_values.items()
            }

        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "seed": self.seed,
            "q_table": serialized_q_table,
        }

    def save(self, output_path: str | Path) -> None:
        """
        Sauvegarde l'agent dans un fichier JSON.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_serializable(),
                file,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, input_path: str | Path) -> "QLearningAgent":
        """
        Recharge un agent Q-learning depuis un fichier JSON.
        """

        input_path = Path(input_path)

        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        agent = cls(
            learning_rate=data["learning_rate"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
            seed=data.get("seed"),
        )

        agent.epsilon_start = data.get("epsilon_start", data["epsilon"])

        for state_key, action_values in data["q_table"].items():
            state = tuple(int(value) for value in state_key.split("|"))

            for action, q_value in action_values.items():
                agent.q_table[state][int(action)] = float(q_value)

        return agent