import re
from pathlib import Path
from typing import Dict, Tuple

from src.cvrp.data.instance import CVRPInstance


def _parse_vehicle_count_from_name(name: str) -> int | None:
    """
    Extrait le nombre de véhicules à partir d'un nom comme A-n32-k5.
    Ici, k5 signifie 5 véhicules.
    """
    match = re.search(r"-k(\d+)", name.lower())
    if match:
        return int(match.group(1))
    return None


def _clean_key_value(line: str) -> tuple[str, str] | None:
    """
    Lit les lignes de type :
    NAME : A-n32-k5
    DIMENSION : 32
    CAPACITY : 100
    """
    if ":" not in line:
        return None

    key, value = line.split(":", 1)
    return key.strip().upper(), value.strip()


def load_cvrp_instance(
    file_path: str | Path,
    reference_cost: float | None = None,
) -> CVRPInstance:
    """
    Charge une instance CVRP au format CVRPLIB.

    Sections attendues :
    - NODE_COORD_SECTION
    - DEMAND_SECTION
    - DEPOT_SECTION
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    name: str | None = None
    dimension: int | None = None
    capacity: int | None = None
    edge_weight_type = "EUC_2D"

    coordinates: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    depot: int | None = None

    section: str | None = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            upper_line = line.upper()

            if upper_line == "EOF":
                break

            if upper_line == "NODE_COORD_SECTION":
                section = "NODE_COORD_SECTION"
                continue

            if upper_line == "DEMAND_SECTION":
                section = "DEMAND_SECTION"
                continue

            if upper_line == "DEPOT_SECTION":
                section = "DEPOT_SECTION"
                continue

            key_value = _clean_key_value(line)
            if key_value and section is None:
                key, value = key_value

                if key == "NAME":
                    name = value

                elif key == "DIMENSION":
                    dimension = int(value)

                elif key == "CAPACITY":
                    capacity = int(value)

                elif key == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = value

                continue

            parts = line.split()

            if section == "NODE_COORD_SECTION":
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates[node_id] = (x, y)

            elif section == "DEMAND_SECTION":
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(float(parts[1]))
                    demands[node_id] = demand

            elif section == "DEPOT_SECTION":
                depot_candidate = int(parts[0])
                if depot_candidate == -1:
                    section = None
                    continue
                depot = depot_candidate

    if name is None:
        name = file_path.stem

    if dimension is None:
        dimension = len(coordinates)

    if capacity is None:
        raise ValueError("CAPACITY introuvable dans le fichier.")

    if depot is None:
        raise ValueError("DEPOT_SECTION introuvable ou dépôt non défini.")

    if len(coordinates) != dimension:
        raise ValueError(
            f"Incohérence dimension : dimension={dimension}, "
            f"coordonnées lues={len(coordinates)}"
        )

    for node in coordinates:
        if node not in demands:
            demands[node] = 0

    vehicle_count = _parse_vehicle_count_from_name(name)

    return CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        vehicle_count=vehicle_count,
        depot=depot,
        coordinates=coordinates,
        demands=demands,
        edge_weight_type=edge_weight_type,
        reference_cost=reference_cost,
    )