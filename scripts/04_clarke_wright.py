from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from cvrp.data import load_cvrplib_instance
from cvrp.core import (
    route_demand,
    route_cost,
    solution_cost,
    validate_solution,
    compute_gap,
)
from cvrp.heuristics.nearest_neighbor import nearest_neighbor_cvrp
from cvrp.heuristics.two_opt import two_opt_solution
from cvrp.heuristics.clarke_wright import clarke_wright_savings


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"
REFERENCE_COST = 784


def summarize_solution(method_name, solution, instance, elapsed_time):
    valid, errors = validate_solution(solution, instance)
    total_cost = solution_cost(solution, instance)
    gap = compute_gap(total_cost, instance.reference_cost)

    print(f"=== {method_name} ===")
    print("Solution valide :", valid)
    print("Nombre de routes :", len(solution))
    print("Coût total :", total_cost)
    print("Gap (%) :", gap)
    print("Temps :", round(elapsed_time, 6), "seconde(s)")

    if errors:
        print("Erreurs :")
        for error in errors:
            print("-", error)

    print()

    return {
        "method": method_name,
        "cost": total_cost,
        "gap": gap,
        "routes": len(solution),
        "valid": valid,
        "time": elapsed_time,
    }


def print_routes(solution, instance):
    for index, route in enumerate(solution, start=1):
        print(f"Route {index} : {route}")
        print(f"  Demande : {route_demand(route, instance)} / {instance.capacity}")
        print(f"  Coût    : {route_cost(route, instance)}")
        print()


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

    results = []

    # 1. Nearest Neighbor CVRP
    start = time.perf_counter()
    nn_solution = nearest_neighbor_cvrp(instance, max_vehicles=None)
    elapsed = time.perf_counter() - start

    results.append(
        summarize_solution(
            method_name="Nearest Neighbor CVRP",
            solution=nn_solution,
            instance=instance,
            elapsed_time=elapsed,
        )
    )

    # 2. Nearest Neighbor + 2-opt
    start = time.perf_counter()
    nn_two_opt_solution = two_opt_solution(nn_solution, instance)
    elapsed = time.perf_counter() - start

    results.append(
        summarize_solution(
            method_name="Nearest Neighbor + 2-opt",
            solution=nn_two_opt_solution,
            instance=instance,
            elapsed_time=elapsed,
        )
    )

    # 3. Clarke & Wright Savings
    start = time.perf_counter()
    cw_solution = clarke_wright_savings(instance)
    elapsed = time.perf_counter() - start

    results.append(
        summarize_solution(
            method_name="Clarke & Wright Savings",
            solution=cw_solution,
            instance=instance,
            elapsed_time=elapsed,
        )
    )

    # 4. Clarke & Wright + 2-opt
    start = time.perf_counter()
    cw_two_opt_solution = two_opt_solution(cw_solution, instance)
    elapsed = time.perf_counter() - start

    results.append(
        summarize_solution(
            method_name="Clarke & Wright + 2-opt",
            solution=cw_two_opt_solution,
            instance=instance,
            elapsed_time=elapsed,
        )
    )

    print("=== ROUTES CLARKE & WRIGHT + 2-OPT ===")
    print_routes(cw_two_opt_solution, instance)

    print("=== TABLEAU COMPARATIF ===")
    print("| Méthode | Coût | Gap (%) | Routes | Faisable | Temps (s) |")
    print("|---|---:|---:|---:|---|---:|")

    for result in results:
        print(
            f"| {result['method']} "
            f"| {result['cost']} "
            f"| {result['gap']} "
            f"| {result['routes']} "
            f"| {'Oui' if result['valid'] else 'Non'} "
            f"| {round(result['time'], 6)} |"
        )


if __name__ == "__main__":
    main()