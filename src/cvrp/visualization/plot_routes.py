from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from src.cvrp.data.instance import CVRPInstance
from src.cvrp.core.validation import validate_solution


def plot_solution_routes(
    instance: CVRPInstance,
    routes: List[List[int]],
    title: str,
    output_path: str | Path,
    show_node_labels: bool = True,
    show_route_labels: bool = True,
) -> None:
    """
    Génère une figure représentant les routes d'une solution CVRP.

    La figure permet d'analyser :
    - la forme des tournées ;
    - les croisements éventuels ;
    - la compacité des routes ;
    - les détours ;
    - l'effet d'une amélioration locale comme 2-opt.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validation = validate_solution(instance, routes)

    fig, ax = plt.subplots(figsize=(9, 7))

    depot = instance.depot
    depot_x, depot_y = instance.coordinates[depot]

    # Dessiner les clients.
    customer_x = []
    customer_y = []

    for customer in instance.customers:
        x, y = instance.coordinates[customer]
        customer_x.append(x)
        customer_y.append(y)

    ax.scatter(customer_x, customer_y, marker="o", label="Clients")
    ax.scatter([depot_x], [depot_y], marker="s", s=120, label="Dépôt")

    # Dessiner les routes.
    for route_index, route in enumerate(routes, start=1):
        x_values = []
        y_values = []

        for node in route:
            x, y = instance.coordinates[node]
            x_values.append(x)
            y_values.append(y)

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.5,
            label=f"Route {route_index}" if show_route_labels else None,
        )

    # Étiquettes des noeuds.
    if show_node_labels:
        for node, (x, y) in instance.coordinates.items():
            if node == depot:
                ax.annotate(
                    f"D{node}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                demand = instance.demands.get(node, 0)
                ax.annotate(
                    f"{node}\nq={demand}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    cost_text = "Coût : N/A"
    gap_text = "Gap : N/A"

    if validation.cost is not None:
        cost_text = f"Coût : {validation.cost:.2f}"

    if validation.gap_percent is not None:
        gap_text = f"Gap : {validation.gap_percent:.2f} %"

    validity_text = "Valide" if validation.is_valid else "Invalide"

    subtitle = (
        f"{instance.name} | {cost_text} | {gap_text} | "
        f"Routes : {len(routes)} | {validity_text}"
    )

    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("Coordonnée X")
    ax.set_ylabel("Coordonnée Y")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_route_analysis(
    instance: CVRPInstance,
    routes: List[List[int]],
    method_name: str,
) -> None:
    """
    Affiche une analyse textuelle simple d'une solution.

    Cette sortie servira plus tard à rédiger l'analyse cartographique
    dans le mémoire.
    """

    validation = validate_solution(instance, routes)

    print("\n" + "=" * 80)
    print(method_name)
    print("=" * 80)

    print("Valide :", validation.is_valid)
    print("Coût :", validation.cost)
    print("Gap (%) :", validation.gap_percent)
    print("Nombre de routes :", len(routes))
    print("Charges :", validation.route_loads)

    for index, route in enumerate(routes, start=1):
        load = instance.route_demand(route)
        cost = instance.route_cost(route)

        print(
            f"Route {index} | charge={load}/{instance.capacity} | "
            f"coût={cost:.2f} | clients={route[1:-1]}"
        )

    if validation.errors:
        print("\nErreurs :")
        for error in validation.errors:
            print("-", error)

    if validation.warnings:
        print("\nWarnings :")
        for warning in validation.warnings:
            print("-", warning)