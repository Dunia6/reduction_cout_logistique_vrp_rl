import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.core.validation import validate_solution
from src.cvrp.heuristics.nearest_neighbor import nearest_neighbor_cvrp
from src.cvrp.heuristics.clarke_wright import clarke_wright_savings
from src.cvrp.heuristics.two_opt import two_opt_solution
from src.cvrp.visualization.plot_routes import (
    plot_solution_routes,
    print_route_analysis,
)


REFERENCE_COSTS = {
    "A-n32-k5": 784,
}


def method_slug(method_name: str) -> str:
    """
    Transforme le nom d'une méthode en nom de fichier propre.
    """
    return (
        method_name.lower()
        .replace("&", "and")
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("-", "_")
    )


def serialize_solution(instance, method_name, routes):
    """
    Prépare les routes pour sauvegarde JSON.
    """
    validation = validate_solution(instance, routes)

    return {
        "instance": instance.name,
        "method": method_name,
        "cost": validation.cost,
        "gap_percent": validation.gap_percent,
        "is_valid": validation.is_valid,
        "routes_count": len(routes),
        "route_loads": validation.route_loads,
        "routes": routes,
        "errors": validation.errors,
        "warnings": validation.warnings,
    }


def main():
    instance_path = Path("data/instances/A-n32-k5.vrp")

    if not instance_path.exists():
        raise FileNotFoundError(
            f"Instance introuvable : {instance_path}. "
            "Place le fichier A-n32-k5.vrp dans data/instances/."
        )

    instance = load_cvrp_instance(
        instance_path,
        reference_cost=REFERENCE_COSTS.get(instance_path.stem),
    )

    output_dir = Path("results/heuristics/routes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construction des solutions.
    nn_routes = nearest_neighbor_cvrp(instance)
    nn_2opt_routes = two_opt_solution(instance, nn_routes)

    cw_routes = clarke_wright_savings(instance)
    cw_2opt_routes = two_opt_solution(instance, cw_routes)

    methods = [
        ("Nearest Neighbor CVRP", nn_routes),
        ("Nearest Neighbor + 2-opt", nn_2opt_routes),
        ("Clarke & Wright Savings", cw_routes),
        ("Clarke & Wright + 2-opt", cw_2opt_routes),
    ]

    serialized_solutions = []

    for method_name, routes in methods:
        print_route_analysis(instance, routes, method_name)

        slug = method_slug(method_name)
        output_path = output_dir / f"{instance.name}_{slug}.png"

        plot_solution_routes(
            instance=instance,
            routes=routes,
            title=method_name,
            output_path=output_path,
            show_node_labels=True,
            show_route_labels=True,
        )

        serialized_solutions.append(
            serialize_solution(instance, method_name, routes)
        )

        print("Figure générée :", output_path)

    json_output_path = output_dir / f"{instance.name}_heuristic_routes.json"

    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(serialized_solutions, file, indent=2, ensure_ascii=False)

    print("\nRoutes sauvegardées :", json_output_path)


if __name__ == "__main__":
    main()
