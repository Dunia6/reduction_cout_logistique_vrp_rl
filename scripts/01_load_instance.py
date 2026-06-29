from pathlib import Path
import sys

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


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"

# Valeur de référence connue pour A-n32-k5.
# CVRPLIB indique 784 pour cette instance.
REFERENCE_COST = 784


def main():
    instance = load_cvrplib_instance(INSTANCE_PATH)
    instance.reference_cost = REFERENCE_COST

    print("=== INSTANCE CHARGÉE ===")
    print("Nom :", instance.name)
    print("Nombre de nœuds :", len(instance.demands))
    print("Nombre de clients :", len(instance.demands) - 1)
    print("Dépôt :", instance.depot)
    print("Capacité véhicule :", instance.capacity)
    print("Nombre de véhicules :", instance.vehicle_count)
    print("Valeur de référence :", instance.reference_cost)
    print()

    print("=== PREMIERS NŒUDS ===")
    for node_id in range(min(5, len(instance.demands))):
        x, y = instance.coords[node_id]
        demand = instance.demands[node_id]
        print(f"Nœud {node_id} | x={x} | y={y} | demande={demand}")

    print()

    # Solution de test naïve.
    # Attention : ce n'est pas une vraie bonne solution.
    # Elle sert seulement à vérifier que le calcul fonctionne.
    solution = [
        [0, 1, 2, 3, 4, 5, 0],
        [0, 6, 7, 8, 9, 10, 0],
        [0, 11, 12, 13, 14, 15, 0],
        [0, 16, 17, 18, 19, 20, 0],
        [0, 21, 22, 23, 24, 25, 0],
        [0, 26, 27, 28, 29, 30, 31, 0],
    ]

    print("=== TEST SOLUTION MANUELLE ===")

    for idx, route in enumerate(solution, start=1):
        print(f"Route {idx} : {route}")
        print("  Demande :", route_demand(route, instance))
        print("  Coût    :", route_cost(route, instance))
        print()

    valid, errors = validate_solution(solution, instance)
    total_cost = solution_cost(solution, instance)

    print("Solution valide :", valid)

    if not valid:
        print("Erreurs :")
        for error in errors:
            print("-", error)

    print("Coût total :", total_cost)

    if instance.reference_cost:
        print("Gap (%) :", compute_gap(total_cost, instance.reference_cost))


if __name__ == "__main__":
    main()