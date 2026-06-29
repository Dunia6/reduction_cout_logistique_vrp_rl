from pathlib import Path
import sys
import json
import time

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
from cvrp.heuristics.two_opt import two_opt_solution
from cvrp.visualization import plot_solution


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"

BEST_SOLUTION_PATH = ROOT_DIR / "results" / "masked_policy_rl_best_solution.json"
EXPERIMENTS_PATH = ROOT_DIR / "results" / "experiments.csv"
FIGURE_PATH = ROOT_DIR / "results" / "figures" / "routes_masked_policy_rl_two_opt.png"

REFERENCE_COST = 784


def append_result_to_csv(instance, solution, elapsed_time):
    valid, errors = validate_solution(solution, instance)
    cost = solution_cost(solution, instance)
    gap = compute_gap(cost, instance.reference_cost)

    row = {
        "instance": instance.name,
        "method": "Masked Policy RL + 2-opt",
        "cost": cost,
        "reference_cost": instance.reference_cost,
        "gap_percent": gap,
        "routes": len(solution),
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
                & (df["method"] == "Masked Policy RL + 2-opt")
            )
        ]

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    else:
        df = pd.DataFrame([row])

    df.to_csv(EXPERIMENTS_PATH, index=False)

    return row


def main():
    instance = load_cvrplib_instance(INSTANCE_PATH)
    instance.reference_cost = REFERENCE_COST

    if not BEST_SOLUTION_PATH.exists():
        raise FileNotFoundError(
            "Le fichier masked_policy_rl_best_solution.json est introuvable. "
            "Exécute d'abord scripts/09d_train_masked_policy_rl.py"
        )

    data = json.loads(BEST_SOLUTION_PATH.read_text(encoding="utf-8"))

    rl_solution = data["solution"]

    initial_cost = solution_cost(rl_solution, instance)
    initial_gap = compute_gap(initial_cost, instance.reference_cost)

    print("=== SOLUTION MASKED POLICY RL INITIALE ===")
    print("Coût :", initial_cost)
    print("Gap (%) :", initial_gap)
    print("Routes :", len(rl_solution))
    print()

    start_time = time.perf_counter()
    improved_solution = two_opt_solution(rl_solution, instance)
    elapsed_time = time.perf_counter() - start_time

    valid, errors = validate_solution(improved_solution, instance)
    improved_cost = solution_cost(improved_solution, instance)
    improved_gap = compute_gap(improved_cost, instance.reference_cost)

    improvement = initial_cost - improved_cost
    improvement_percent = round((improvement / initial_cost) * 100, 2)

    print("=== SOLUTION MASKED POLICY RL + 2-OPT ===")
    print("Solution valide :", valid)
    print("Coût :", improved_cost)
    print("Gap (%) :", improved_gap)
    print("Routes :", len(improved_solution))
    print("Amélioration absolue :", round(improvement, 2))
    print("Amélioration (%) :", improvement_percent)
    print("Temps 2-opt :", round(elapsed_time, 6), "seconde(s)")

    if errors:
        print("Erreurs :")
        for error in errors:
            print("-", error)

    row = append_result_to_csv(
        instance=instance,
        solution=improved_solution,
        elapsed_time=elapsed_time,
    )

    print()
    print("Ligne ajoutée au CSV :")
    print(row)

    plot_solution(
        solution=improved_solution,
        instance=instance,
        output_path=FIGURE_PATH,
        title="Routes obtenues par Masked Policy RL + 2-opt",
        show_node_labels=True,
    )


if __name__ == "__main__":
    main()