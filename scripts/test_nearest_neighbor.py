from pprint import pprint
from time import perf_counter

from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.heuristics.nearest_neighbor import nearest_neighbor_cvrp
from src.cvrp.core.validation import validate_solution
from src.cvrp.core.metrics import build_final_result_row
from src.cvrp.paths import INSTANCES_DIR


def main():
    instance = load_cvrp_instance(
        INSTANCES_DIR / "A-n32-k5.vrp",
        reference_cost=784,
    )

    start = perf_counter()
    routes = nearest_neighbor_cvrp(instance)
    inference_time = perf_counter() - start

    validation = validate_solution(instance, routes)

    print("Instance :", instance.name)
    print("Solution valide :", validation.is_valid)
    print("Coût :", validation.cost)
    print("Gap (%) :", validation.gap_percent)
    print("Nombre de routes :", len(routes))
    print("Charges :", validation.route_loads)

    print("\nRoutes :")
    for idx, route in enumerate(routes, start=1):
        print(f"Route {idx} :", route)

    if validation.errors:
        print("\nErreurs :")
        for error in validation.errors:
            print("-", error)

    row = build_final_result_row(
        instance=instance,
        method="Nearest Neighbor CVRP",
        category="heuristic_baseline",
        routes=routes,
        seed=None,
        train_time_sec=0.0,
        inference_time_sec=inference_time,
        postprocess_time_sec=0.0,
        episodes=None,
        notes="Heuristique constructive simple du plus proche voisin.",
    )

    print("\nLigne standardisée pour les résultats :")
    pprint(row)


if __name__ == "__main__":
    main()
