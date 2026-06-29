# VRP-RL — Mémoire CVRP

Projet de recherche sur le **Capacitated Vehicle Routing Problem (CVRP)** : comparaison entre heuristiques classiques et approches par **apprentissage par renforcement** (Q-learning et politique masquée).

Instance principale : **A-n32-k5** (CVRPLIB, coût de référence : 784).

## Structure du projet

```
vrp-rl-memoire/
├── data/instances/          # Instances CVRPLIB (.vrp)
├── scripts/                 # Scripts d'expérimentation (ordre numéroté)
├── src/cvrp/
│   ├── core.py              # Coûts, validation, gap
│   ├── data.py              # Chargement des instances
│   ├── visualization.py     # Tracé des routes
│   ├── heuristics/          # Nearest Neighbor, Clarke-Wright, 2-opt
│   └── rl/                  # Environnement RL, agents Q-learning et politique masquée
└── results/
    ├── experiments.csv      # Résultats comparatifs
    ├── summary_table.md     # Tableau Markdown pour le mémoire
    └── figures/             # Graphiques et visualisations de routes
```

## Installation

Python 3.12 recommandé.

```bash
cd vrp-rl-memoire
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

> **PyTorch** : si l'installation via `requirements.txt` échoue selon votre plateforme, installez d'abord PyTorch depuis [pytorch.org](https://pytorch.org), puis les autres dépendances.

## Méthodes comparées

| Méthode | Script principal |
|---|---|
| Nearest Neighbor CVRP | `02_nearest_neighbor_cvrp.py` |
| Nearest Neighbor + 2-opt | `3_nearest_neighbor_two_opt.py` |
| Clarke & Wright Savings | `04_clarke_wright.py` |
| Clarke & Wright + 2-opt | `07_visualize_routes.py` |
| Q-learning RL | `09_train_q_learning.py` |
| Masked Policy RL | `09d_train_masked_policy_rl.py` |
| Masked Policy RL + 2-opt | `12_masked_policy_rl_two_opt.py` |

## Exécution

Depuis la racine du projet, avec l'environnement virtuel activé :

```bash
# 1. Vérifier le chargement de l'instance
python scripts/01_load_instance.py

# 2. Heuristiques (tests unitaires)
python scripts/02_nearest_neighbor_cvrp.py
python scripts/3_nearest_neighbor_two_opt.py
python scripts/04_clarke_wright.py

# 3. Benchmark heuristiques → results/experiments.csv
python scripts/05_run_experiments.py

# 4. Graphiques comparatifs (barres, gap, temps)
python scripts/06_generate_charts.py

# 5. Visualisation des routes heuristiques
python scripts/07_visualize_routes.py

# 6. Environnement RL (test)
python scripts/08_test_rl_env.py

# 7. Entraînement RL (ajoutent leurs lignes à experiments.csv)
python scripts/09_train_q_learning.py
python scripts/09d_train_masked_policy_rl.py   # peut prendre plusieurs minutes

# 8. Post-traitement RL + 2-opt
python scripts/12_masked_policy_rl_two_opt.py

# 9. Régénérer les graphiques après toutes les expériences
python scripts/06_generate_charts.py
```

## Sorties générées

| Fichier | Description |
|---|---|
| `results/experiments.csv` | Coût, gap, routes, temps par méthode |
| `results/summary_table.md` | Tableau récapitulatif Markdown |
| `results/figures/cost_by_method.png` | Coût total par méthode |
| `results/figures/gap_by_method.png` | Écart à la référence (%) |
| `results/figures/routes_by_method.png` | Nombre de routes |
| `results/figures/time_by_method.png` | Temps d'exécution |
| `results/figures/routes_*.png` | Cartes des routes par méthode |
| `results/rl_best_solution.json` | Meilleure solution Q-learning |
| `results/masked_policy_rl_best_solution.json` | Meilleure solution politique masquée |

## Dépendances principales

- **numpy**, **pandas** — calculs et tableaux de résultats
- **matplotlib** — graphiques et visualisation des routes
- **torch** — réseau de politique masquée (Masked Policy RL)
- **vrplib** — lecture des instances CVRPLIB
- **tabulate** — export Markdown (`pandas.to_markdown`)
