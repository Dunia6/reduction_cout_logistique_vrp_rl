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
from cvrp.heuristics.nearest_neighbor import nearest_neighbor_cvrp
from cvrp.heuristics.clarke_wright import clarke_wright_savings
from cvrp.heuristics.two_opt import two_opt_solution
from cvrp.visualization import plot_solution


INSTANCE_PATH = ROOT_DIR / "data" / "instances" / "A-n32-k5.vrp"
FIGURES_DIR = ROOT_DIR / "results" / "figures"

REFERENCE_COST = 784

ROUTE_FIGURES = [
    {
        "filename": "routes_nearest_neighbor.png",
        "title": "Routes obtenues par Nearest Neighbor CVRP",
        "method": "Nearest Neighbor CVRP",
    },
    {
        "filename": "routes_nearest_neighbor_two_opt.png",
        "title": "Routes obtenues par Nearest Neighbor + 2-opt",
        "method": "Nearest Neighbor + 2-opt",
    },
    {
        "filename": "routes_clarke_wright.png",
        "title": "Routes obtenues par Clarke & Wright Savings",
        "method": "Clarke & Wright Savings",
    },
    {
        "filename": "routes_clarke_wright_two_opt.png",
        "title": "Routes obtenues par Clarke & Wright + 2-opt",
        "method": "Clarke & Wright + 2-opt",
    },
]


def print_routes(title: str, solution, instance) -> None:
    print(title)
    for index, route in enumerate(solution, start=1):
        print(f"Route {index} : {route}")
        print(f"  Demande : {route_demand(route, instance)} / {instance.capacity}")
        print(f"  Coût    : {route_cost(route, instance)}")
        print()


def summarize_and_plot(
    *,
    method_name: str,
    solution,
    instance,
    output_path: Path,
    title: str,
) -> None:
    valid, errors = validate_solution(solution, instance)
    total_cost = solution_cost(solution, instance)
    gap = compute_gap(total_cost, instance.reference_cost)

    print_routes(f"=== ROUTES {method_name.upper()} ===", solution, instance)

    print("=== RÉSUMÉ ===")
    print("Méthode :", method_name)
    print("Solution valide :", valid)
    print("Nombre de routes :", len(solution))
    print("Coût total :", total_cost)
    print("Gap (%) :", gap)

    if errors:
        print("Erreurs :")
        for error in errors:
            print("-", error)

    print()

    plot_solution(
        solution=solution,
        instance=instance,
        output_path=output_path,
        title=title,
        show_node_labels=True,
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

    nn_solution = nearest_neighbor_cvrp(instance, max_vehicles=None)
    nn_two_opt_solution = two_opt_solution(nn_solution, instance)
    cw_solution = clarke_wright_savings(instance)
    cw_two_opt_solution = two_opt_solution(cw_solution, instance)

    solutions = {
        "Nearest Neighbor CVRP": nn_solution,
        "Nearest Neighbor + 2-opt": nn_two_opt_solution,
        "Clarke & Wright Savings": cw_solution,
        "Clarke & Wright + 2-opt": cw_two_opt_solution,
    }

    for figure in ROUTE_FIGURES:
        summarize_and_plot(
            method_name=figure["method"],
            solution=solutions[figure["method"]],
            instance=instance,
            output_path=FIGURES_DIR / figure["filename"],
            title=figure["title"],
        )

    print("=== FICHIERS GÉNÉRÉS ===")
    for figure in ROUTE_FIGURES:
        output_path = FIGURES_DIR / figure["filename"]
        status = "OK" if output_path.exists() else "MANQUANT"
        print(f"[{status}] {output_path.resolve()}")


if __name__ == "__main__":
    main()
