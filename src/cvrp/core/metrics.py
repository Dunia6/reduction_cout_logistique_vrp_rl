from typing import Any, Dict

from src.cvrp.data.instance import CVRPInstance
from src.cvrp.core.validation import validate_solution


FINAL_RESULT_COLUMNS = [
    "instance",
    "method",
    "category",
    "seed",
    "cost",
    "reference_cost",
    "gap_percent",
    "routes",
    "vehicle_count",
    "is_valid",
    "missing_customers_count",
    "duplicated_customers_count",
    "capacity_violations_count",
    "train_time_sec",
    "inference_time_sec",
    "postprocess_time_sec",
    "total_time_sec",
    "episodes",
    "notes",
]


def build_final_result_row(
    instance: CVRPInstance,
    method: str,
    category: str,
    routes: list[list[int]],
    seed: int | None = None,
    train_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
    postprocess_time_sec: float = 0.0,
    episodes: int | None = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Construit une ligne standardisée de résultats.

    Cette fonction doit être utilisée par :
    - les heuristiques ;
    - Q-learning ;
    - POMO ;
    - les méthodes hybrides avec 2-opt.

    Objectif :
    garantir que toutes les méthodes produisent les mêmes métriques finales.
    """

    if routes is None:
        routes = []

    validation = validate_solution(
        instance=instance,
        routes=routes,
        require_all_customers=True,
        check_vehicle_count=True,
    )

    capacity_violations_count = sum(
        1 for load in validation.route_loads if load > instance.capacity
    )

    total_time_sec = train_time_sec + inference_time_sec + postprocess_time_sec

    return {
        "instance": instance.name,
        "method": method,
        "category": category,
        "seed": seed,
        "cost": validation.cost,
        "reference_cost": instance.reference_cost,
        "gap_percent": validation.gap_percent,
        "routes": len(routes),
        "vehicle_count": instance.vehicle_count,
        "is_valid": validation.is_valid,
        "missing_customers_count": len(validation.missing_customers),
        "duplicated_customers_count": len(validation.duplicated_customers),
        "capacity_violations_count": capacity_violations_count,
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "postprocess_time_sec": postprocess_time_sec,
        "total_time_sec": total_time_sec,
        "episodes": episodes,
        "notes": notes,
    }