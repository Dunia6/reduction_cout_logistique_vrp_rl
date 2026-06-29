from pathlib import Path
import sys
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
from cvrp.heuristics.nearest_neighbor import nearest_neighbor_cvrp
from cvrp.heuristics.two_opt import two_opt_solution
from cvrp.heuristics.clarke_wright import clarke_wright_savings


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"
RESULTS_PATH = ROOT_DIR / "results" / "experiments.csv"

REFERENCE_COSTS = {
    "A-n32-k5": 784,
}


def evaluate_method(method_name, build_solution_fn, instance):
    """
    Exécute une méthode, mesure son temps, valide la solution,
    calcule son coût et son gap.
    """
    start_time = time.perf_counter()

    solution = build_solution_fn()

    elapsed_time = time.perf_counter() - start_time

    valid, errors = validate_solution(solution, instance)
    cost = solution_cost(solution, instance)

    reference_cost = instance.reference_cost
    gap = compute_gap(cost, reference_cost) if reference_cost else None

    return {
        "instance": instance.name,
        "method": method_name,
        "cost": cost,
        "reference_cost": reference_cost,
        "gap_percent": gap,
        "routes": len(solution),
        "vehicle_reference": instance.vehicle_count,
        "valid": valid,
        "time_seconds": round(elapsed_time, 6),
        "errors": " | ".join(errors) if errors else "",
    }


def main():
    instance = load_cvrplib_instance(INSTANCE_PATH)

    instance.reference_cost = REFERENCE_COSTS.get(instance.name)

    print("=== INSTANCE ===")
    print("Nom :", instance.name)
    print("Clients :", len(instance.demands) - 1)
    print("Capacité :", instance.capacity)
    print("Véhicules indiqués :", instance.vehicle_count)
    print("Référence :", instance.reference_cost)
    print()

    results = []

    # 1. Nearest Neighbor
    nn_solution = None

    def run_nearest_neighbor():
        nonlocal nn_solution
        nn_solution = nearest_neighbor_cvrp(instance, max_vehicles=None)
        return nn_solution

    results.append(
        evaluate_method(
            method_name="Nearest Neighbor CVRP",
            build_solution_fn=run_nearest_neighbor,
            instance=instance,
        )
    )

    # 2. Nearest Neighbor + 2-opt
    def run_nearest_neighbor_two_opt():
        base_solution = nn_solution or nearest_neighbor_cvrp(instance, max_vehicles=None)
        return two_opt_solution(base_solution, instance)

    results.append(
        evaluate_method(
            method_name="Nearest Neighbor + 2-opt",
            build_solution_fn=run_nearest_neighbor_two_opt,
            instance=instance,
        )
    )

    # 3. Clarke & Wright
    cw_solution = None

    def run_clarke_wright():
        nonlocal cw_solution
        cw_solution = clarke_wright_savings(instance)
        return cw_solution

    results.append(
        evaluate_method(
            method_name="Clarke & Wright Savings",
            build_solution_fn=run_clarke_wright,
            instance=instance,
        )
    )

    # 4. Clarke & Wright + 2-opt
    def run_clarke_wright_two_opt():
        base_solution = cw_solution or clarke_wright_savings(instance)
        return two_opt_solution(base_solution, instance)

    results.append(
        evaluate_method(
            method_name="Clarke & Wright + 2-opt",
            build_solution_fn=run_clarke_wright_two_opt,
            instance=instance,
        )
    )

    df = pd.DataFrame(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    print("=== RÉSULTATS ===")
    print(df.to_string(index=False))
    print()
    print(f"Résultats sauvegardés dans : {RESULTS_PATH}")


if __name__ == "__main__":
    main()