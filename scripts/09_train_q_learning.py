from pathlib import Path
import sys
import json
import time
import random

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from cvrp.data import load_cvrplib_instance
from cvrp.core import (
    solution_cost,
    validate_solution,
    compute_gap,
)
from cvrp.rl.cvrp_env import CVRPEnv
from cvrp.rl.q_learning_agent import QLearningAgent
from cvrp.visualization import plot_solution


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"

RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENTS_PATH = RESULTS_DIR / "experiments.csv"
BEST_SOLUTION_PATH = RESULTS_DIR / "rl_best_solution.json"
RL_ROUTES_FIGURE_PATH = RESULTS_DIR / "figures" / "routes_q_learning.png"

REFERENCE_COST = 784


def run_training_episode(
    env: CVRPEnv,
    agent: QLearningAgent,
    seed: int | None = None,
) -> dict:
    """
    Exécute un épisode d'entraînement complet.
    """
    observation = env.reset(seed=seed)
    total_reward = 0.0

    while not env.done:
        state_key = agent.get_state_key(observation)

        action = agent.choose_action(env, observation)

        next_observation, reward, done, info = env.step(action)

        next_state_key = None if done else agent.get_state_key(next_observation)
        next_valid_actions = [] if done else env.get_valid_actions()

        agent.update(
            state_key=state_key,
            action=action,
            reward=reward,
            next_state_key=next_state_key,
            next_valid_actions=next_valid_actions,
            done=done,
        )

        total_reward += reward
        observation = next_observation

    agent.decay_epsilon()

    summary = env.summary()
    summary["total_reward"] = round(total_reward, 2)

    return summary


def evaluate_agent_greedy(
    instance,
    agent: QLearningAgent,
) -> dict:
    """
    Évalue l'agent sans exploration.

    On force epsilon = 0 pour utiliser uniquement ce que l'agent a appris.
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    env = CVRPEnv(
        instance=instance,
        max_routes=None,
        invalid_action_penalty=-1000.0,
        incomplete_solution_penalty=-5000.0,
        completion_bonus=100.0,
    )

    observation = env.reset(seed=123)

    while not env.done:
        action = agent.choose_action(env, observation)
        observation, reward, done, info = env.step(action)

    agent.epsilon = old_epsilon

    return env.summary()


def append_rl_result_to_csv(
    instance,
    best_summary: dict,
    elapsed_time: float,
) -> None:
    """
    Ajoute ou remplace la ligne RL dans results/experiments.csv.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    valid, errors = validate_solution(best_summary["solution"], instance)
    cost = solution_cost(best_summary["solution"], instance)
    gap = compute_gap(cost, instance.reference_cost)

    row = {
        "instance": instance.name,
        "method": "Q-learning RL",
        "cost": cost,
        "reference_cost": instance.reference_cost,
        "gap_percent": gap,
        "routes": len(best_summary["solution"]),
        "vehicle_reference": instance.vehicle_count,
        "valid": valid,
        "time_seconds": round(elapsed_time, 6),
        "errors": " | ".join(errors) if errors else "",
    }

    if EXPERIMENTS_PATH.exists():
        df = pd.read_csv(EXPERIMENTS_PATH)

        # Supprimer l'ancienne ligne Q-learning RL si elle existe déjà.
        df = df[
            ~(
                (df["instance"] == instance.name)
                & (df["method"] == "Q-learning RL")
            )
        ]

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    else:
        df = pd.DataFrame([row])

    df.to_csv(EXPERIMENTS_PATH, index=False)

    print(f"Résultat RL enregistré dans : {EXPERIMENTS_PATH}")


def save_best_solution(best_summary: dict, instance) -> None:
    """
    Sauvegarde la meilleure solution RL dans un fichier JSON.
    """
    data = {
        "instance": instance.name,
        "reference_cost": instance.reference_cost,
        "solution": best_summary["solution"],
        "cost": best_summary["cost"],
        "routes": best_summary["routes"],
        "valid": best_summary["valid"],
        "errors": best_summary["errors"],
        "gap_percent": compute_gap(best_summary["cost"], instance.reference_cost),
        "steps": best_summary["steps"],
        "invalid_actions": best_summary["invalid_actions"],
    }

    BEST_SOLUTION_PATH.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    print(f"Meilleure solution RL sauvegardée dans : {BEST_SOLUTION_PATH}")


def main():
    random.seed(42)
    np.random.seed(42)

    instance = load_cvrplib_instance(INSTANCE_PATH)
    instance.reference_cost = REFERENCE_COST

    print("=== INSTANCE ===")
    print("Nom :", instance.name)
    print("Clients :", len(instance.demands) - 1)
    print("Dépôt :", instance.depot)
    print("Capacité :", instance.capacity)
    print("Véhicules indiqués :", instance.vehicle_count)
    print("Référence :", instance.reference_cost)
    print()

    env = CVRPEnv(
        instance=instance,
        max_routes=None,
        invalid_action_penalty=-1000.0,
        incomplete_solution_penalty=-5000.0,
        completion_bonus=100.0,
    )

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    episodes = 1000

    best_summary = None
    best_cost = float("inf")

    start_time = time.perf_counter()

    print("=== ENTRAÎNEMENT Q-LEARNING ===")

    for episode in range(1, episodes + 1):
        summary = run_training_episode(
            env=env,
            agent=agent,
            seed=episode,
        )

        if summary["valid"] and summary["cost"] < best_cost:
            best_cost = summary["cost"]
            best_summary = summary

        if episode % 100 == 0:
            current_cost = summary["cost"]
            current_gap = compute_gap(current_cost, instance.reference_cost)

            print(
                f"Épisode {episode}/{episodes} | "
                f"Coût courant={current_cost} | "
                f"Meilleur={best_cost} | "
                f"Gap courant={current_gap}% | "
                f"Epsilon={round(agent.epsilon, 4)}"
            )

    elapsed_time = time.perf_counter() - start_time

    print()
    print("=== ÉVALUATION GREEDY DE L'AGENT ===")

    greedy_summary = evaluate_agent_greedy(instance, agent)

    if greedy_summary["valid"] and greedy_summary["cost"] < best_cost:
        best_summary = greedy_summary
        best_cost = greedy_summary["cost"]

    if best_summary is None:
        print("Aucune solution valide trouvée par l'agent RL.")
        return

    final_gap = compute_gap(best_summary["cost"], instance.reference_cost)

    print("Solution :", best_summary["solution"])
    print("Valide :", best_summary["valid"])
    print("Coût :", best_summary["cost"])
    print("Routes :", best_summary["routes"])
    print("Gap (%) :", final_gap)
    print("Étapes :", best_summary["steps"])
    print("Actions invalides :", best_summary["invalid_actions"])
    print("Temps total entraînement :", round(elapsed_time, 4), "seconde(s)")
    print()

    save_best_solution(best_summary, instance)

    append_rl_result_to_csv(
        instance=instance,
        best_summary=best_summary,
        elapsed_time=elapsed_time,
    )

    plot_solution(
        solution=best_summary["solution"],
        instance=instance,
        output_path=RL_ROUTES_FIGURE_PATH,
        title="Routes obtenues par Q-learning RL",
        show_node_labels=True,
    )


if __name__ == "__main__":
    main()