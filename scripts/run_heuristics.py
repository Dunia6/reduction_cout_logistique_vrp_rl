import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.paths import INSTANCES_DIR, RESULTS_DIR
from src.cvrp.core.metrics import FINAL_RESULT_COLUMNS, build_final_result_row
from src.cvrp.core.validation import validate_solution
from src.cvrp.heuristics.nearest_neighbor import nearest_neighbor_cvrp
from src.cvrp.heuristics.clarke_wright import clarke_wright_savings
from src.cvrp.heuristics.two_opt import two_opt_solution


# Références connues CVRPLIB.
# Ajoute ici les nouvelles instances quand tu vas passer au multi-instances.
REFERENCE_COSTS: Dict[str, float] = {
    "A-n32-k5": 784,
    "A-n37-k6": 669,
    "A-n39-k6": 831,

}


def get_reference_cost(instance_path: Path) -> float | None:
    """
    Retourne le coût de référence si l'instance est connue.
    Sinon retourne None.
    """
    return REFERENCE_COSTS.get(instance_path.stem)


def run_single_instance(instance_path: Path) -> List[dict]:
    """
    Exécute toutes les heuristiques principales sur une instance CVRP.

    Méthodes exécutées :
    - Nearest Neighbor ;
    - Nearest Neighbor + 2-opt ;
    - Clarke & Wright Savings ;
    - Clarke & Wright + 2-opt.
    """

    reference_cost = get_reference_cost(instance_path)

    instance = load_cvrp_instance(
        instance_path,
        reference_cost=reference_cost,
    )

    rows: List[dict] = []

    print("\n" + "=" * 80)
    print(f"Instance : {instance.name}")
    print(f"Fichier  : {instance_path}")
    print(f"Clients  : {instance.customer_count}")
    print(f"Véhicules: {instance.vehicle_count}")
    print(f"Capacité : {instance.capacity}")
    print(f"Référence: {instance.reference_cost}")
    print("=" * 80)

    # ---------------------------------------------------------------------
    # 1. Nearest Neighbor
    # ---------------------------------------------------------------------
    start = perf_counter()
    nn_routes = nearest_neighbor_cvrp(instance)
    nn_inference_time = perf_counter() - start

    nn_validation = validate_solution(instance, nn_routes)

    rows.append(
        build_final_result_row(
            instance=instance,
            method="Nearest Neighbor CVRP",
            category="heuristic_baseline",
            routes=nn_routes,
            seed=None,
            train_time_sec=0.0,
            inference_time_sec=nn_inference_time,
            postprocess_time_sec=0.0,
            episodes=None,
            notes="Heuristique constructive simple du plus proche voisin.",
        )
    )

    print_result(
        method="Nearest Neighbor CVRP",
        validation=nn_validation,
        inference_time=nn_inference_time,
        postprocess_time=0.0,
    )

    # ---------------------------------------------------------------------
    # 2. Nearest Neighbor + 2-opt
    # ---------------------------------------------------------------------
    start = perf_counter()
    nn_2opt_routes = two_opt_solution(instance, nn_routes)
    nn_2opt_postprocess_time = perf_counter() - start

    nn_2opt_validation = validate_solution(instance, nn_2opt_routes)

    rows.append(
        build_final_result_row(
            instance=instance,
            method="Nearest Neighbor + 2-opt",
            category="heuristic_local_search",
            routes=nn_2opt_routes,
            seed=None,
            train_time_sec=0.0,
            inference_time_sec=nn_inference_time,
            postprocess_time_sec=nn_2opt_postprocess_time,
            episodes=None,
            notes="Nearest Neighbor amélioré par 2-opt intra-route.",
        )
    )

    print_result(
        method="Nearest Neighbor + 2-opt",
        validation=nn_2opt_validation,
        inference_time=nn_inference_time,
        postprocess_time=nn_2opt_postprocess_time,
    )

    # ---------------------------------------------------------------------
    # 3. Clarke & Wright Savings
    # ---------------------------------------------------------------------
    start = perf_counter()
    cw_routes = clarke_wright_savings(instance)
    cw_inference_time = perf_counter() - start

    cw_validation = validate_solution(instance, cw_routes)

    rows.append(
        build_final_result_row(
            instance=instance,
            method="Clarke & Wright Savings",
            category="heuristic_constructive",
            routes=cw_routes,
            seed=None,
            train_time_sec=0.0,
            inference_time_sec=cw_inference_time,
            postprocess_time_sec=0.0,
            episodes=None,
            notes="Heuristique constructive classique basée sur les économies.",
        )
    )

    print_result(
        method="Clarke & Wright Savings",
        validation=cw_validation,
        inference_time=cw_inference_time,
        postprocess_time=0.0,
    )

    # ---------------------------------------------------------------------
    # 4. Clarke & Wright + 2-opt
    # ---------------------------------------------------------------------
    start = perf_counter()
    cw_2opt_routes = two_opt_solution(instance, cw_routes)
    cw_2opt_postprocess_time = perf_counter() - start

    cw_2opt_validation = validate_solution(instance, cw_2opt_routes)

    rows.append(
        build_final_result_row(
            instance=instance,
            method="Clarke & Wright + 2-opt",
            category="heuristic_local_search",
            routes=cw_2opt_routes,
            seed=None,
            train_time_sec=0.0,
            inference_time_sec=cw_inference_time,
            postprocess_time_sec=cw_2opt_postprocess_time,
            episodes=None,
            notes="Clarke & Wright amélioré par 2-opt intra-route.",
        )
    )

    print_result(
        method="Clarke & Wright + 2-opt",
        validation=cw_2opt_validation,
        inference_time=cw_inference_time,
        postprocess_time=cw_2opt_postprocess_time,
    )

    return rows


def print_result(
    method: str,
    validation,
    inference_time: float,
    postprocess_time: float,
) -> None:
    """
    Affiche un résumé lisible d'une méthode.
    """

    total_time = inference_time + postprocess_time

    print("\n---", method, "---")
    print("Valide              :", validation.is_valid)
    print("Coût                :", validation.cost)
    print("Gap (%)             :", validation.gap_percent)
    print("Routes              :", len(validation.route_loads))
    print("Charges             :", validation.route_loads)
    print("Temps inférence (s) :", inference_time)
    print("Temps 2-opt (s)     :", postprocess_time)
    print("Temps total (s)     :", total_time)

    if validation.errors:
        print("Erreurs :")
        for error in validation.errors:
            print(" -", error)

    if validation.warnings:
        print("Warnings :")
        for warning in validation.warnings:
            print(" -", warning)


def write_results_csv(rows: List[dict], output_path: Path) -> None:
    """
    Écrit les résultats dans un CSV standardisé.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FINAL_RESULT_COLUMNS,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\nCSV généré :", output_path)


def main():
    instances_dir = INSTANCES_DIR
    output_path = RESULTS_DIR / "heuristics/experiments_heuristics.csv"

    if not instances_dir.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {instances_dir}. "
            "Place les fichiers .vrp dans data/instances/."
        )

    instance_files = sorted(instances_dir.glob("*.vrp"))

    if not instance_files:
        raise FileNotFoundError(
            "Aucun fichier .vrp trouvé dans data/instances/."
        )

    all_rows: List[dict] = []

    for instance_path in instance_files:
        try:
            rows = run_single_instance(instance_path)
            all_rows.extend(rows)
        except Exception as exc:
            print("\nErreur sur l'instance :", instance_path)
            print("Message :", exc)

    write_results_csv(all_rows, output_path)

    print("\nNombre total de lignes :", len(all_rows))


if __name__ == "__main__":
    main()
