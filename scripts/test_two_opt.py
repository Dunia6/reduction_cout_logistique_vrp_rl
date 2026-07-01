from pprint import pprint
from time import perf_counter

from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.heuristics.nearest_neighbor import nearest_neighbor_cvrp
from src.cvrp.heuristics.two_opt import two_opt_solution, two_opt_gain
from src.cvrp.core.validation import validate_solution
from src.cvrp.core.metrics import build_final_result_row
from src.cvrp.paths import INSTANCES_DIR


def main():
    instance = load_cvrp_instance(
        INSTANCES_DIR / "A-n32-k5.vrp",
        reference_cost=784,
    )

    # 1. Construction initiale avec Nearest Neighbor
    start = perf_counter()
    nn_routes = nearest_neighbor_cvrp(instance)
    nn_time = perf_counter() - start

    nn_validation = validate_solution(instance, nn_routes)

    # 2. Amélioration locale avec 2-opt
    start = perf_counter()
    nn_2opt_routes = two_opt_solution(instance, nn_routes)
    postprocess_time = perf_counter() - start

    nn_2opt_validation = validate_solution(instance, nn_2opt_routes)

    before_cost, after_cost, gain = two_opt_gain(
        instance,
        nn_routes,
        nn_2opt_routes,
    )

    print("Instance :", instance.name)

    print("\n--- Nearest Neighbor ---")
    print("Valide :", nn_validation.is_valid)
    print("Coût :", nn_validation.cost)
    print("Gap (%) :", nn_validation.gap_percent)
    print("Routes :", len(nn_routes))
    print("Charges :", nn_validation.route_loads)
    print("Temps inférence :", nn_time)

    print("\n--- Nearest Neighbor + 2-opt ---")
    print("Valide :", nn_2opt_validation.is_valid)
    print("Coût :", nn_2opt_validation.cost)
    print("Gap (%) :", nn_2opt_validation.gap_percent)
    print("Routes :", len(nn_2opt_routes))
    print("Charges :", nn_2opt_validation.route_loads)
    print("Temps post-traitement :", postprocess_time)

    print("\n--- Gain 2-opt ---")
    print("Coût avant :", before_cost)
    print("Coût après :", after_cost)
    print("Gain :", gain)

    print("\nRoutes après 2-opt :")
    for idx, route in enumerate(nn_2opt_routes, start=1):
        print(f"Route {idx} :", route)

    row_nn = build_final_result_row(
        instance=instance,
        method="Nearest Neighbor CVRP",
        category="heuristic_baseline",
        routes=nn_routes,
        seed=None,
        train_time_sec=0.0,
        inference_time_sec=nn_time,
        postprocess_time_sec=0.0,
        episodes=None,
        notes="Heuristique constructive simple du plus proche voisin.",
    )

    row_nn_2opt = build_final_result_row(
        instance=instance,
        method="Nearest Neighbor + 2-opt",
        category="heuristic_local_search",
        routes=nn_2opt_routes,
        seed=None,
        train_time_sec=0.0,
        inference_time_sec=nn_time,
        postprocess_time_sec=postprocess_time,
        episodes=None,
        notes="Nearest Neighbor amélioré par 2-opt intra-route.",
    )

    print("\nLigne standardisée Nearest Neighbor :")
    pprint(row_nn)

    print("\nLigne standardisée Nearest Neighbor + 2-opt :")
    pprint(row_nn_2opt)


if __name__ == "__main__":
    main()
