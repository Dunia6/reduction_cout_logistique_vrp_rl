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


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"
REFERENCE_COST = 784


def print_solution(title, solution, instance):
    print(title)

    for index, route in enumerate(solution, start=1):
        demand = route_demand(route, instance)
        cost = route_cost(route, instance)

        print(f"Route {index} : {route}")
        print(f"  Demande : {demand} / {instance.capacity}")
        print(f"  Coût    : {cost}")
        print()

    valid, errors = validate_solution(solution, instance)
    total_cost = solution_cost(solution, instance)

    print("Solution valide :", valid)
    print("Nombre de routes :", len(solution))
    print("Coût total :", total_cost)

    if instance.reference_cost:
        print("Gap (%) :", compute_gap(total_cost, instance.reference_cost))

    if errors:
        print("Erreurs :")
        for error in errors:
            print("-", error)

    print("-" * 60)
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

    start_nn = time.perf_counter()
    nn_solution = nearest_neighbor_cvrp(
        instance=instance,
        max_vehicles=None,
    )
    nn_time = time.perf_counter() - start_nn

    start_two_opt = time.perf_counter()
    improved_solution = two_opt_solution(nn_solution, instance)
    two_opt_time = time.perf_counter() - start_two_opt

    print_solution(
        title="=== SOLUTION NEAREST NEIGHBOR CVRP ===",
        solution=nn_solution,
        instance=instance,
    )

    print_solution(
        title="=== SOLUTION NEAREST NEIGHBOR + 2-OPT ===",
        solution=improved_solution,
        instance=instance,
    )

    nn_cost = solution_cost(nn_solution, instance)
    improved_cost = solution_cost(improved_solution, instance)

    improvement = nn_cost - improved_cost
    improvement_percent = round((improvement / nn_cost) * 100, 2) if nn_cost > 0 else 0

    print("=== COMPARAISON ===")
    print("Coût Nearest Neighbor :", nn_cost)
    print("Temps Nearest Neighbor :", round(nn_time, 6), "seconde(s)")
    print()
    print("Coût NN + 2-opt :", improved_cost)
    print("Temps 2-opt :", round(two_opt_time, 6), "seconde(s)")
    print()
    print("Amélioration absolue :", round(improvement, 2))
    print("Amélioration (%) :", improvement_percent)

    if instance.reference_cost:
        print()
        print("Gap NN (%) :", compute_gap(nn_cost, instance.reference_cost))
        print("Gap NN + 2-opt (%) :", compute_gap(improved_cost, instance.reference_cost))


if __name__ == "__main__":
    main()