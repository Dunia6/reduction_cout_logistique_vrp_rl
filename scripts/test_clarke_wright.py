from pprint import pprint
from time import perf_counter

from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.heuristics.clarke_wright import clarke_wright_savings
from src.cvrp.heuristics.two_opt import two_opt_solution, two_opt_gain
from src.cvrp.core.validation import validate_solution
from src.cvrp.core.metrics import build_final_result_row
from src.cvrp.paths import INSTANCES_DIR


def main():
    instance = load_cvrp_instance(
        INSTANCES_DIR / "A-n32-k5.vrp",
        reference_cost=784,
    )

    # 1. Construction Clarke & Wright
    start = perf_counter()
    cw_routes = clarke_wright_savings(instance)
    cw_time = perf_counter() - start

    cw_validation = validate_solution(instance, cw_routes)

    # 2. Amélioration Clarke & Wright + 2-opt
    start = perf_counter()
    cw_2opt_routes = two_opt_solution(instance, cw_routes)
    postprocess_time = perf_counter() - start

    cw_2opt_validation = validate_solution(instance, cw_2opt_routes)

    before_cost, after_cost, gain = two_opt_gain(
        instance,
        cw_routes,
        cw_2opt_routes,
    )

    print("Instance :", instance.name)

    print("\n--- Clarke & Wright Savings ---")
    print("Valide :", cw_validation.is_valid)
    print("Coût :", cw_validation.cost)
    print("Gap (%) :", cw_validation.gap_percent)
    print("Routes :", len(cw_routes))
    print("Charges :", cw_validation.route_loads)
    print("Temps inférence :", cw_time)

    print("\nRoutes Clarke & Wright :")
    for idx, route in enumerate(cw_routes, start=1):
        print(f"Route {idx} :", route)

    if cw_validation.errors:
        print("\nErreurs Clarke & Wright :")
        for error in cw_validation.errors:
            print("-", error)

    print("\n--- Clarke & Wright + 2-opt ---")
    print("Valide :", cw_2opt_validation.is_valid)
    print("Coût :", cw_2opt_validation.cost)
    print("Gap (%) :", cw_2opt_validation.gap_percent)
    print("Routes :", len(cw_2opt_routes))
    print("Charges :", cw_2opt_validation.route_loads)
    print("Temps post-traitement :", postprocess_time)

    print("\n--- Gain 2-opt ---")
    print("Coût avant :", before_cost)
    print("Coût après :", after_cost)
    print("Gain :", gain)

    print("\nRoutes Clarke & Wright + 2-opt :")
    for idx, route in enumerate(cw_2opt_routes, start=1):
        print(f"Route {idx} :", route)

    row_cw = build_final_result_row(
        instance=instance,
        method="Clarke & Wright Savings",
        category="heuristic_constructive",
        routes=cw_routes,
        seed=None,
        train_time_sec=0.0,
        inference_time_sec=cw_time,
        postprocess_time_sec=0.0,
        episodes=None,
        notes="Heuristique constructive classique basée sur les économies.",
    )

    row_cw_2opt = build_final_result_row(
        instance=instance,
        method="Clarke & Wright + 2-opt",
        category="heuristic_local_search",
        routes=cw_2opt_routes,
        seed=None,
        train_time_sec=0.0,
        inference_time_sec=cw_time,
        postprocess_time_sec=postprocess_time,
        episodes=None,
        notes="Clarke & Wright amélioré par 2-opt intra-route.",
    )

    print("\nLigne standardisée Clarke & Wright :")
    pprint(row_cw)

    print("\nLigne standardisée Clarke & Wright + 2-opt :")
    pprint(row_cw_2opt)


if __name__ == "__main__":
    main()
