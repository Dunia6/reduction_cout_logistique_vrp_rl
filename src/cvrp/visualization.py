from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from cvrp.data import CVRPInstance
from cvrp.core import Solution, route_cost, route_demand, solution_cost


def plot_solution(
    solution: Solution,
    instance: CVRPInstance,
    output_path: str | Path,
    title: str = "Solution CVRP",
    show_node_labels: bool = True,
) -> None:
    """
    Visualise une solution CVRP sur un plan 2D.

    Chaque route est tracée sous forme de ligne :
        dépôt → client → client → dépôt

    L'image est sauvegardée dans output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = instance.coords
    depot = instance.depot

    plt.figure(figsize=(10, 8))

    # Afficher toutes les positions des nœuds.
    x_values = coords[:, 0]
    y_values = coords[:, 1]

    plt.scatter(x_values, y_values, marker="o", label="Clients")

    # Afficher le dépôt avec un marqueur différent.
    depot_x = coords[depot][0]
    depot_y = coords[depot][1]
    plt.scatter([depot_x], [depot_y], marker="s", s=120, label="Dépôt")

    # Afficher les numéros des nœuds.
    if show_node_labels:
        for node_id, (x, y) in enumerate(coords):
            label = str(node_id)
            plt.text(x + 0.5, y + 0.5, label, fontsize=8)

    # Tracer chaque route.
    for route_index, route in enumerate(solution, start=1):
        route_x = [coords[node][0] for node in route]
        route_y = [coords[node][1] for node in route]

        demand = route_demand(route, instance)
        cost = route_cost(route, instance)

        plt.plot(
            route_x,
            route_y,
            marker="o",
            linewidth=1.8,
            label=f"Route {route_index} | q={demand} | c={cost}",
        )

    total_cost = solution_cost(solution, instance)

    plt.title(f"{title}\nCoût total = {total_cost} | Routes = {len(solution)}")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")