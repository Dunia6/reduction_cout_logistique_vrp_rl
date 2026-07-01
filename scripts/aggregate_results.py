from pathlib import Path

import pandas as pd


FINAL_COLUMNS = [
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


INPUT_FILES = [
    Path("results/heuristics/experiments_heuristics.csv"),
    Path("results/q_learning/experiments_q_learning.csv"),
    Path("results/pomo/experiments_pomo.csv"),
]


OUTPUT_DIR = Path("results/final")


def read_existing_results() -> pd.DataFrame:
    """
    Lit tous les fichiers de résultats disponibles.

    Si un fichier n'existe pas encore, il est ignoré.
    Cela permet d'agréger progressivement :
    - heuristiques seules ;
    - heuristiques + Q-learning ;
    - heuristiques + Q-learning + POMO.
    """

    frames = []

    for path in INPUT_FILES:
        if not path.exists():
            print(f"[IGNORÉ] fichier absent : {path}")
            continue

        df = pd.read_csv(path)
        df["source_file"] = str(path)

        print(f"[OK] fichier chargé : {path} | lignes={len(df)}")

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "Aucun fichier de résultats trouvé. "
            "Lance d'abord les heuristiques, Q-learning ou POMO."
        )

    combined = pd.concat(frames, ignore_index=True)

    return combined


def ensure_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit que toutes les colonnes finales existent.

    Les colonnes manquantes sont ajoutées avec des valeurs nulles.
    """

    for column in FINAL_COLUMNS:
        if column not in df.columns:
            df[column] = None

    ordered_columns = FINAL_COLUMNS + [
        col for col in df.columns if col not in FINAL_COLUMNS
    ]

    return df[ordered_columns]


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les types utiles pour les calculs statistiques.
    """

    numeric_columns = [
        "seed",
        "cost",
        "reference_cost",
        "gap_percent",
        "routes",
        "vehicle_count",
        "missing_customers_count",
        "duplicated_customers_count",
        "capacity_violations_count",
        "train_time_sec",
        "inference_time_sec",
        "postprocess_time_sec",
        "total_time_sec",
        "episodes",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "is_valid" in df.columns:
        df["is_valid"] = df["is_valid"].astype(str).str.lower().map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "valid": True,
                "false": False,
                "0": False,
                "no": False,
                "invalid": False,
            }
        ).fillna(df["is_valid"])

    return df


def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des rangs par instance.

    Le rang principal est basé sur le coût moyen par méthode.
    """

    ranked = df.copy()

    method_mean = (
        ranked.groupby(["instance", "method"], dropna=False)
        .agg(mean_cost=("cost", "mean"))
        .reset_index()
    )

    method_mean["rank_by_cost"] = method_mean.groupby("instance")["mean_cost"].rank(
        method="dense",
        ascending=True,
    )

    ranked = ranked.merge(
        method_mean[["instance", "method", "rank_by_cost"]],
        on=["instance", "method"],
        how="left",
    )

    return ranked


def build_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé par instance et méthode.

    Important pour le mémoire :
    - moyenne ;
    - écart-type ;
    - min ;
    - max ;
    - taux de validité ;
    - temps moyens séparés.
    """

    summary = (
        df.groupby(["instance", "method", "category"], dropna=False)
        .agg(
            runs=("method", "count"),
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std"),
            cost_min=("cost", "min"),
            cost_max=("cost", "max"),
            reference_cost=("reference_cost", "first"),
            gap_mean=("gap_percent", "mean"),
            gap_std=("gap_percent", "std"),
            gap_min=("gap_percent", "min"),
            gap_max=("gap_percent", "max"),
            routes_mean=("routes", "mean"),
            valid_rate=("is_valid", "mean"),
            train_time_mean=("train_time_sec", "mean"),
            inference_time_mean=("inference_time_sec", "mean"),
            postprocess_time_mean=("postprocess_time_sec", "mean"),
            total_time_mean=("total_time_sec", "mean"),
            episodes=("episodes", "max"),
        )
        .reset_index()
    )

    summary["rank_by_cost"] = summary.groupby("instance")["cost_mean"].rank(
        method="dense",
        ascending=True,
    )

    summary = summary.sort_values(
        by=["instance", "rank_by_cost", "cost_mean"],
        ascending=[True, True, True],
    )

    return summary


def build_instance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé général par instance.
    """

    valid_df = df[df["is_valid"] == True].copy()

    if valid_df.empty:
        return pd.DataFrame()

    best_rows = (
        valid_df.sort_values(["instance", "cost"], ascending=[True, True])
        .groupby("instance", as_index=False)
        .first()
    )

    instance_summary = best_rows[
        [
            "instance",
            "method",
            "category",
            "cost",
            "reference_cost",
            "gap_percent",
            "routes",
            "vehicle_count",
            "total_time_sec",
        ]
    ].copy()

    instance_summary = instance_summary.rename(
        columns={
            "method": "best_method",
            "category": "best_category",
            "cost": "best_cost",
            "gap_percent": "best_gap_percent",
            "routes": "best_routes",
            "total_time_sec": "best_total_time_sec",
        }
    )

    return instance_summary


def build_tableau_memoire(method_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un tableau simplifié pour insertion dans le mémoire.

    On garde les colonnes lisibles :
    - méthode ;
    - coût moyen ;
    - écart-type ;
    - gap moyen ;
    - validité ;
    - temps total moyen.
    """

    table = method_summary.copy()

    table = table[
        [
            "instance",
            "rank_by_cost",
            "method",
            "category",
            "runs",
            "cost_mean",
            "cost_std",
            "cost_min",
            "cost_max",
            "reference_cost",
            "gap_mean",
            "gap_std",
            "valid_rate",
            "train_time_mean",
            "inference_time_mean",
            "postprocess_time_mean",
            "total_time_mean",
            "episodes",
        ]
    ].copy()

    numeric_rounding = {
        "rank_by_cost": 0,
        "cost_mean": 2,
        "cost_std": 2,
        "cost_min": 2,
        "cost_max": 2,
        "reference_cost": 2,
        "gap_mean": 2,
        "gap_std": 2,
        "valid_rate": 3,
        "train_time_mean": 4,
        "inference_time_mean": 4,
        "postprocess_time_mean": 4,
        "total_time_mean": 4,
    }

    for column, decimals in numeric_rounding.items():
        if column in table.columns:
            table[column] = table[column].round(decimals)

    return table


def save_outputs(
    detailed_df: pd.DataFrame,
    method_summary: pd.DataFrame,
    instance_summary: pd.DataFrame,
    tableau_memoire: pd.DataFrame,
) -> None:
    """
    Sauvegarde tous les fichiers finaux.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = OUTPUT_DIR / "experiments_detailed.csv"
    method_summary_path = OUTPUT_DIR / "method_summary.csv"
    instance_summary_path = OUTPUT_DIR / "instance_summary.csv"
    tableau_path = OUTPUT_DIR / "tableau_memoire_resultats.csv"

    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8")
    method_summary.to_csv(method_summary_path, index=False, encoding="utf-8")
    instance_summary.to_csv(instance_summary_path, index=False, encoding="utf-8")
    tableau_memoire.to_csv(tableau_path, index=False, encoding="utf-8")

    print("\nFichiers générés :")
    print("-", detailed_path)
    print("-", method_summary_path)
    print("-", instance_summary_path)
    print("-", tableau_path)


def print_best_results(method_summary: pd.DataFrame) -> None:
    """
    Affiche les meilleurs résultats par instance.
    """

    print("\nMeilleures méthodes par instance :")

    for instance_name, group in method_summary.groupby("instance"):
        best = group.sort_values("cost_mean").iloc[0]

        print(
            f"- {instance_name} : {best['method']} | "
            f"coût moyen={best['cost_mean']:.2f} | "
            f"gap moyen={best['gap_mean']:.2f}% | "
            f"validité={best['valid_rate']:.2f}"
        )


def main():
    print("Agrégation des résultats expérimentaux")
    print("=" * 80)

    detailed_df = read_existing_results()
    detailed_df = ensure_final_columns(detailed_df)
    detailed_df = clean_types(detailed_df)
    detailed_df = add_rankings(detailed_df)

    method_summary = build_method_summary(detailed_df)
    instance_summary = build_instance_summary(detailed_df)
    tableau_memoire = build_tableau_memoire(method_summary)

    save_outputs(
        detailed_df=detailed_df,
        method_summary=method_summary,
        instance_summary=instance_summary,
        tableau_memoire=tableau_memoire,
    )

    print_best_results(method_summary)

    print("\nAperçu du tableau mémoire :")
    print(tableau_memoire.to_string(index=False))


if __name__ == "__main__":
    main()