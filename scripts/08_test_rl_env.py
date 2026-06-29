from pathlib import Path
import sys
import random

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from cvrp.data import load_cvrplib_instance
from cvrp.core import compute_gap
from cvrp.rl.cvrp_env import CVRPEnv


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"
REFERENCE_COST = 784


def nearest_feasible_action(env: CVRPEnv) -> int:
    """
    Politique simple pour tester l'environnement.

    Elle choisit le client faisable le plus proche.
    Si aucun client n'est faisable, elle retourne au dépôt.
    """
    valid_actions = env.get_valid_actions()

    if not valid_actions:
        return env.depot

    client_actions = [
        action for action in valid_actions
        if action != env.depot
    ]

    if not client_actions:
        return env.depot

    current_node = env.current_node

    return min(
        client_actions,
        key=lambda client: env.instance.distance_matrix[current_node][client],
    )


def random_feasible_action(env: CVRPEnv) -> int:
    """
    Politique aléatoire faisable.

    Elle choisit seulement parmi les actions autorisées.
    """
    valid_actions = env.get_valid_actions()

    if not valid_actions:
        return env.depot

    return random.choice(valid_actions)


def run_episode(env: CVRPEnv, policy_name: str = "nearest") -> dict:
    """
    Exécute un épisode complet avec une politique donnée.
    """
    observation = env.reset(seed=42)

    print("=== ÉTAT INITIAL ===")
    print(observation)
    print()

    step_index = 0

    while not env.done:
        step_index += 1

        if policy_name == "random":
            action = random_feasible_action(env)
        else:
            action = nearest_feasible_action(env)

        observation, reward, done, info = env.step(action)

        print(f"Step {step_index}")
        print("  Action :", action)
        print("  Reward :", round(reward, 2))
        print("  Message :", info["message"])
        print("  Position actuelle :", observation["current_node"])
        print("  Capacité restante :", observation["remaining_capacity"])
        print("  Distance totale :", observation["total_distance"])
        print("  Routes construites :", observation["routes_count"])
        print()

    return env.summary()


def main():
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

    summary = run_episode(env, policy_name="nearest")

    print("=== RÉSUMÉ ÉPISODE ===")
    print("Solution :", summary["solution"])
    print("Valide :", summary["valid"])
    print("Coût :", summary["cost"])
    print("Routes :", summary["routes"])
    print("Steps :", summary["steps"])
    print("Actions invalides :", summary["invalid_actions"])

    if instance.reference_cost:
        print("Gap (%) :", compute_gap(summary["cost"], instance.reference_cost))

    if summary["errors"]:
        print("Erreurs :")
        for error in summary["errors"]:
            print("-", error)


if __name__ == "__main__":
    main()