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


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"
REFERENCE_COST = 784


def print_solution(solution, instance):
    for index, route in enumerate(solution, start=1):
        demand = route_demand(route, instance)
        cost = route_cost(route, instance)

        print(f"Route {index} : {route}")
        print(f"  Demande : {demand} / {instance.capacity}")
        print(f"  Coût    : {cost}")
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

    start_time = time.perf_counter()

    solution = nearest_neighbor_cvrp(
        instance=instance,
        max_vehicles=None,
    )

    elapsed_time = time.perf_counter() - start_time

    print("=== SOLUTION NEAREST NEIGHBOR CVRP ===")
    print_solution(solution, instance)

    valid, errors = validate_solution(solution, instance)
    total_cost = solution_cost(solution, instance)

    print("=== RÉSUMÉ ===")
    print("Solution valide :", valid)
    print("Nombre de routes utilisées :", len(solution))
    print("Coût total :", total_cost)
    print("Temps d'exécution :", round(elapsed_time, 6), "seconde(s)")

    if instance.reference_cost:
        print("Gap (%) :", compute_gap(total_cost, instance.reference_cost))

    if instance.vehicle_count is not None:
        print("Véhicules de référence :", instance.vehicle_count)

        if len(solution) > instance.vehicle_count:
            print(
                "Attention : la solution utilise plus de routes que le nombre de véhicules "
                "indiqué dans l'instance."
            )

    if errors:
        print()
        print("Erreurs détectées :")
        for error in errors:
            print("-", error)


if __name__ == "__main__":
    main()