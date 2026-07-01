# VRP-RL — Mémoire CVRP

Projet de recherche sur le **Capacitated Vehicle Routing Problem (CVRP)** : comparaison entre heuristiques classiques et approches par **apprentissage par renforcement**.

Instances CVRPLIB disponibles dans `data/instances/` (dont **A-n32-k5**, **A-n37-k5**, **A-n39-k6**).

## Méthodes implémentées

| Méthode | Où l'exécuter |
|---|---|
| Nearest Neighbor CVRP | `scripts/run_heuristics.py` |
| Nearest Neighbor + 2-opt | `scripts/run_heuristics.py` |
| Clarke & Wright Savings | `scripts/run_heuristics.py` |
| Clarke & Wright + 2-opt | `scripts/run_heuristics.py` |
| Q-learning RL | `notebooks/01_q_learning_cvrp_experiments.ipynb` |
| POMO-style Active Search (variantes + 2-opt) | `notebooks/02_pomo_active_search_cvrp_experiments.ipynb` |

Toutes les méthodes passent par `build_final_result_row` (`src/cvrp/core/metrics.py`) et partagent les mêmes colonnes de résultats.

Le modèle neuronal du notebook 02 est un **MLP léger** (ReLU, optimiseur **Adam**). Le réseau est défini dans le notebook, pas dans `src/`.

## Structure du projet

```
vrp-rl-memoire/
├── data/instances/          # Fichiers .vrp (CVRPLIB)
├── notebooks/
│   ├── 01_q_learning_cvrp_experiments.ipynb
│   └── 02_pomo_active_search_cvrp_experiments.ipynb
├── scripts/
│   ├── run_heuristics.py
│   ├── aggregate_results.py
│   ├── plot_heuristic_routes.py
│   └── test_*.py
├── src/cvrp/
│   ├── paths.py
│   ├── core/                # cost, validation, metrics
│   ├── data/                # instance, parser
│   ├── heuristics/          # nearest_neighbor, clarke_wright, two_opt
│   ├── rl/                  # env, features, q_learning_agent
│   └── visualization/       # plot_routes
├── results/
│   ├── heuristics/
│   ├── q_learning/
│   ├── pomo/
│   └── final/
└── requirements.txt
```

## Installation

Python **3.12** recommandé.

```bash
cd vrp-rl-memoire
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> **PyTorch** : en cas d'échec, installer d'abord depuis [pytorch.org](https://pytorch.org), puis relancer `pip install -r requirements.txt`.

Les scripts ajoutent la racine du projet à `PYTHONPATH`. Sous VS Code / Cursor, `.vscode/settings.json` configure aussi `PYTHONPATH`.

## Exécution

```bash
# Tests unitaires
python scripts/test_parser.py
python scripts/test_nearest_neighbor.py
python scripts/test_clarke_wright.py
python scripts/test_two_opt.py
python scripts/test_rl_env.py
python scripts/test_q_learning_agent.py

# Heuristiques → results/heuristics/experiments_heuristics.csv
python scripts/run_heuristics.py

# Visualisation des routes heuristiques
python scripts/plot_heuristic_routes.py

# RL (notebooks Jupyter)
jupyter notebook notebooks/

# Agrégation → results/final/
python scripts/aggregate_results.py
```

`aggregate_results.py` fusionne les CSV présents parmi :

- `results/heuristics/experiments_heuristics.csv`
- `results/q_learning/experiments_q_learning.csv`
- `results/pomo/experiments_pomo.csv`

Les fichiers absents sont ignorés.

## Sorties principales

| Dossier / fichier | Contenu |
|---|---|
| `results/heuristics/experiments_heuristics.csv` | 4 heuristiques |
| `results/q_learning/experiments_q_learning.csv` | Q-learning |
| `results/pomo/experiments_pomo.csv` | POMO-style Active Search |
| `results/final/experiments_detailed.csv` | Fusion de toutes les méthodes |
| `results/final/method_summary.csv` | Statistiques par méthode |
| `results/final/tableau_memoire_resultats.csv` | Tableau simplifié |

Colonnes communes : `instance`, `method`, `category`, `seed`, `cost`, `reference_cost`, `gap_percent`, `routes`, `vehicle_count`, `is_valid`, `missing_customers_count`, `duplicated_customers_count`, `capacity_violations_count`, `train_time_sec`, `inference_time_sec`, `postprocess_time_sec`, `total_time_sec`, `episodes`, `notes`.

## Dépendances

| Paquet | Usage |
|---|---|
| numpy, pandas | Calculs et CSV |
| matplotlib | Graphiques et routes |
| torch | Notebook POMO-style (réseau de politique) |
| vrplib | Instances CVRPLIB |
| tabulate | Export Markdown |
