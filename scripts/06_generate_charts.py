from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT_DIR / "results" / "experiments.csv"
FIGURES_DIR = ROOT_DIR / "results" / "figures"


def load_results() -> pd.DataFrame:
    """
    Charge le fichier CSV des expérimentations.
    """
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Le fichier {RESULTS_PATH} est introuvable. "
            "Exécute d'abord scripts/05_run_experiments.py"
        )

    df = pd.read_csv(RESULTS_PATH)

    required_columns = {
        "instance",
        "method",
        "cost",
        "reference_cost",
        "gap_percent",
        "routes",
        "valid",
        "time_seconds",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing_columns}")

    return df


def _format_value(value: float, kind: str) -> str:
    if kind == "int":
        return f"{int(value)}"
    if kind == "percent":
        return f"{value:.2f}"
    if kind == "time":
        if value < 0.01:
            return f"{value:.6f}"
        if value < 1:
            return f"{value:.4f}"
        return f"{value:.2f}"
    return f"{value:.2f}"


def _annotate_bar_values(
    ax: plt.Axes,
    bars,
    values: Sequence[float],
    fmt: Callable[[float], str],
) -> None:
    """
    Affiche la valeur au-dessus de chaque barre, y compris lorsque la barre
    est trop petite pour être visible à l'échelle du graphique.
    """
    numeric_values = [float(v) for v in values]
    ymax = max(numeric_values) if numeric_values else 0.0
    visibility_threshold = ymax * 0.01 if ymax > 0 else 0.0
    small_bar_index = 0

    for bar, value in zip(bars, numeric_values, strict=True):
        height = value
        if height < visibility_threshold and ymax > 0:
            label_y = visibility_threshold * (0.4 + 0.18 * small_bar_index)
            small_bar_index += 1
        else:
            label_y = height

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            fmt(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if ymax > 0:
        top = ymax * 1.12
        if small_bar_index > 0:
            top = max(top, visibility_threshold * (0.4 + 0.18 * small_bar_index))
        ax.set_ylim(top=top)


def _plot_bar_chart(
    methods: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    value_kind: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values)
    _annotate_bar_values(
        ax,
        bars,
        values,
        fmt=lambda v: _format_value(v, value_kind),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    return fig


def save_cost_chart(df: pd.DataFrame) -> None:
    """
    Génère le graphique du coût total par méthode.
    """
    fig = _plot_bar_chart(
        df["method"],
        df["cost"],
        title="Coût total obtenu par méthode",
        xlabel="Méthode",
        ylabel="Coût total",
        value_kind="float",
    )

    output_path = FIGURES_DIR / "cost_by_method.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Graphique sauvegardé : {output_path}")


def save_gap_chart(df: pd.DataFrame) -> None:
    """
    Génère le graphique du gap par méthode.
    """
    fig = _plot_bar_chart(
        df["method"],
        df["gap_percent"],
        title="Gap par rapport à la valeur de référence",
        xlabel="Méthode",
        ylabel="Gap (%)",
        value_kind="percent",
    )

    output_path = FIGURES_DIR / "gap_by_method.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Graphique sauvegardé : {output_path}")


def save_routes_chart(df: pd.DataFrame) -> None:
    """
    Génère le graphique du nombre de routes utilisées par méthode.
    """
    fig = _plot_bar_chart(
        df["method"],
        df["routes"],
        title="Nombre de routes utilisées par méthode",
        xlabel="Méthode",
        ylabel="Nombre de routes",
        value_kind="int",
    )

    output_path = FIGURES_DIR / "routes_by_method.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Graphique sauvegardé : {output_path}")


def save_execution_time_chart(df: pd.DataFrame) -> None:
    """
    Génère le graphique du temps d'exécution par méthode.
    """
    fig = _plot_bar_chart(
        df["method"],
        df["time_seconds"],
        title="Temps d'exécution par méthode",
        xlabel="Méthode",
        ylabel="Temps d'exécution (secondes)",
        value_kind="time",
    )

    output_path = FIGURES_DIR / "time_by_method.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Graphique sauvegardé : {output_path}")


def save_summary_table(df: pd.DataFrame) -> None:
    """
    Génère un tableau Markdown directement exploitable dans le mémoire.
    """
    summary_columns = [
        "instance",
        "method",
        "cost",
        "reference_cost",
        "gap_percent",
        "routes",
        "valid",
        "time_seconds",
    ]

    summary = df[summary_columns].copy()

    summary["valid"] = pd.Series(summary["valid"]).replace(
        {True: "Oui", False: "Non"}
    )

    markdown_table = summary.to_markdown(index=False)

    output_path = ROOT_DIR / "results" / "summary_table.md"
    output_path.write_text(markdown_table, encoding="utf-8")

    print(f"Tableau Markdown sauvegardé : {output_path}")


def print_generated_outputs() -> None:
    """
    Affiche clairement tous les fichiers produits par ce script.
    """
    output_files = [
        FIGURES_DIR / "cost_by_method.png",
        FIGURES_DIR / "gap_by_method.png",
        FIGURES_DIR / "routes_by_method.png",
        FIGURES_DIR / "time_by_method.png",
        ROOT_DIR / "results" / "summary_table.md",
    ]

    print("=== FICHIERS GÉNÉRÉS ===")
    print(f"Dossier des graphiques : {FIGURES_DIR.resolve()}")
    print(f"Dossier des résultats  : {(ROOT_DIR / 'results').resolve()}")
    print()

    for output_path in output_files:
        status = "OK" if output_path.exists() else "MANQUANT"
        print(f"[{status}] {output_path.resolve()}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    print("=== RÉSULTATS CHARGÉS ===")
    print(df.to_string(index=False))
    print()

    save_cost_chart(df)
    save_gap_chart(df)
    save_routes_chart(df)
    save_execution_time_chart(df)
    save_summary_table(df)

    print()
    print("Génération des graphiques terminée.")
    print()
    print_generated_outputs()


if __name__ == "__main__":
    main()