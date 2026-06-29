from pathlib import Path
import sys
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
from cvrp.rl.masked_policy_agent import (
    MaskedPolicyNetwork,
    PolicyStep,
    masked_categorical_sample,
    masked_categorical_greedy,
)
from cvrp.rl.rl_features import build_node_features, build_action_mask
from cvrp.visualization import plot_solution


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"

RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENTS_PATH = RESULTS_DIR / "experiments.csv"
BEST_SOLUTION_PATH = RESULTS_DIR / "masked_policy_rl_best_solution.json"
RL_ROUTES_FIGURE_PATH = RESULTS_DIR / "figures" / "routes_masked_policy_rl.png"

REFERENCE_COST = 784

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_reward(reward: float) -> float:
    """
    Normalisation simple pour stabiliser l'apprentissage.
    """
    return reward / 100.0


def run_training_episode(
    instance,
    model: MaskedPolicyNetwork,
    gamma: float = 0.99,
) -> tuple[dict, list[PolicyStep]]:
    """
    Exécute un épisode d'entraînement.

    L'agent ne peut choisir que parmi les actions valides.
    """
    env = CVRPEnv(
        instance=instance,
        max_routes=instance.vehicle_count,
        invalid_action_penalty=-2000.0,
        incomplete_solution_penalty=-8000.0,
        completion_bonus=500.0,
    )

    observation = env.reset()
    steps: list[PolicyStep] = []

    while not env.done:
        node_features = build_node_features(env).to(DEVICE)
        action_mask = build_action_mask(env).to(DEVICE)

        # Cas impossible : aucune action valide alors que l'épisode n'est pas fini.
        if not action_mask.any():
            # On arrête l'épisode avec une forte pénalité artificielle.
            fake_log_prob = torch.tensor(0.0, device=DEVICE)
            fake_value = torch.tensor(0.0, device=DEVICE)
            fake_entropy = torch.tensor(0.0, device=DEVICE)

            steps.append(
                PolicyStep(
                    log_prob=fake_log_prob,
                    value=fake_value,
                    reward=normalize_reward(-5000.0),
                    entropy=fake_entropy,
                )
            )
            env.done = True
            break

        logits, value = model(node_features, action_mask)

        action, log_prob, entropy = masked_categorical_sample(logits)

        observation, reward, done, info = env.step(action)

        steps.append(
            PolicyStep(
                log_prob=log_prob,
                value=value,
                reward=normalize_reward(reward),
                entropy=entropy,
            )
        )

    summary = env.summary()

    return summary, steps


def compute_actor_critic_loss(
    steps: list[PolicyStep],
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    Calcule la loss Actor-Critic.

    actor loss : encourage les actions ayant un avantage positif.
    value loss : entraîne l'estimation de valeur.
    entropy    : encourage l'exploration.
    """
    returns = []
    discounted_return = 0.0

    for step in reversed(steps):
        discounted_return = step.reward + gamma * discounted_return
        returns.insert(0, discounted_return)

    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

    if len(returns_tensor) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std() + 1e-8
        )

    policy_losses = []
    value_losses = []
    entropy_terms = []

    for step, return_value in zip(steps, returns_tensor):
        advantage = return_value - step.value.detach()

        policy_losses.append(-step.log_prob * advantage)
        value_losses.append(F.mse_loss(step.value, return_value))
        entropy_terms.append(step.entropy)

    policy_loss = torch.stack(policy_losses).sum()
    value_loss = torch.stack(value_losses).sum()
    entropy_bonus = torch.stack(entropy_terms).sum()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    return loss


def evaluate_greedy(instance, model: MaskedPolicyNetwork) -> dict:
    """
    Évalue le modèle sans échantillonnage.
    À chaque étape, on choisit l'action valide avec le score maximal.
    """
    env = CVRPEnv(
        instance=instance,
        max_routes=instance.vehicle_count,
        invalid_action_penalty=-2000.0,
        incomplete_solution_penalty=-8000.0,
        completion_bonus=500.0,
    )

    env.reset()

    model.eval()

    with torch.no_grad():
        while not env.done:
            node_features = build_node_features(env).to(DEVICE)
            action_mask = build_action_mask(env).to(DEVICE)

            if not action_mask.any():
                env.done = True
                break

            logits, value = model(node_features, action_mask)
            action = masked_categorical_greedy(logits)

            env.step(action)

    model.train()

    return env.summary()


def append_result_to_csv(instance, best_summary: dict, elapsed_time: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    valid, errors = validate_solution(best_summary["solution"], instance)
    cost = solution_cost(best_summary["solution"], instance)
    gap = compute_gap(cost, instance.reference_cost)

    row = {
        "instance": instance.name,
        "method": "Masked Policy RL",
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

        df = df[
            ~(
                (df["instance"] == instance.name)
                & (df["method"] == "Masked Policy RL")
            )
        ]

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    else:
        df = pd.DataFrame([row])

    df.to_csv(EXPERIMENTS_PATH, index=False)

    print(f"Résultat Masked Policy RL enregistré dans : {EXPERIMENTS_PATH}")


def save_best_solution(best_summary: dict, instance) -> None:
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
    torch.manual_seed(42)

    instance = load_cvrplib_instance(INSTANCE_PATH)
    instance.reference_cost = REFERENCE_COST

    print("=== INSTANCE ===")
    print("Nom :", instance.name)
    print("Clients :", len(instance.demands) - 1)
    print("Dépôt :", instance.depot)
    print("Capacité :", instance.capacity)
    print("Véhicules indiqués :", instance.vehicle_count)
    print("Référence :", instance.reference_cost)
    print("Device :", DEVICE)
    print()

    model = MaskedPolicyNetwork(
        node_feature_dim=9,
        hidden_dim=128,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    episodes = 5000
    gamma = 0.99

    best_summary = None
    best_cost = float("inf")

    start_time = time.perf_counter()

    print("=== ENTRAÎNEMENT MASKED POLICY RL ===")

    for episode in range(1, episodes + 1):
        summary, steps = run_training_episode(
            instance=instance,
            model=model,
            gamma=gamma,
        )

        if steps:
            loss = compute_actor_critic_loss(
                steps=steps,
                gamma=gamma,
                value_coef=0.5,
                entropy_coef=0.01,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        valid = summary["valid"]
        routes_ok = summary["routes"] <= instance.vehicle_count

        if valid and routes_ok and summary["cost"] < best_cost:
            best_cost = summary["cost"]
            best_summary = summary

        if episode % 250 == 0:
            greedy_summary = evaluate_greedy(instance, model)

            if (
                greedy_summary["valid"]
                and greedy_summary["routes"] <= instance.vehicle_count
                and greedy_summary["cost"] < best_cost
            ):
                best_cost = greedy_summary["cost"]
                best_summary = greedy_summary

            current_cost = summary["cost"]
            current_gap = compute_gap(current_cost, instance.reference_cost)

            print(
                f"Épisode {episode}/{episodes} | "
                f"Coût courant={current_cost} | "
                f"Meilleur={best_cost} | "
                f"Gap courant={current_gap}% | "
                f"Routes={summary['routes']} | "
                f"Valide={summary['valid']}"
            )

    elapsed_time = time.perf_counter() - start_time

    print()
    print("=== ÉVALUATION FINALE GREEDY ===")

    greedy_summary = evaluate_greedy(instance, model)

    if (
        greedy_summary["valid"]
        and greedy_summary["routes"] <= instance.vehicle_count
        and greedy_summary["cost"] < best_cost
    ):
        best_summary = greedy_summary
        best_cost = greedy_summary["cost"]

    if best_summary is None:
        print("Aucune solution valide trouvée avec le nombre de véhicules autorisé.")
        print("Augmente le nombre d'épisodes ou vérifie le masque d'actions.")
        return

    final_gap = compute_gap(best_summary["cost"], instance.reference_cost)

    print("Solution :", best_summary["solution"])
    print("Valide :", best_summary["valid"])
    print("Coût :", best_summary["cost"])
    print("Routes :", best_summary["routes"])
    print("Gap (%) :", final_gap)
    print("Étapes :", best_summary["steps"])
    print("Actions invalides :", best_summary["invalid_actions"])
    print("Temps total :", round(elapsed_time, 4), "seconde(s)")
    print()

    save_best_solution(best_summary, instance)

    append_result_to_csv(
        instance=instance,
        best_summary=best_summary,
        elapsed_time=elapsed_time,
    )

    plot_solution(
        solution=best_summary["solution"],
        instance=instance,
        output_path=RL_ROUTES_FIGURE_PATH,
        title="Routes obtenues par Masked Policy RL",
        show_node_labels=True,
    )


if __name__ == "__main__":
    main()