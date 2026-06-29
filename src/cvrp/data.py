from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import vrplib


@dataclass
class CVRPInstance:
    name: str
    capacity: int
    depot: int
    coords: np.ndarray
    demands: np.ndarray
    distance_matrix: np.ndarray
    vehicle_count: int | None = None
    reference_cost: float | None = None


def parse_vehicle_count_from_name(name: str) -> int | None:
    """
    Extrait le nombre de véhicules depuis un nom comme A-n32-k5.
    Ici, k5 signifie 5 véhicules.
    """
    match = re.search(r"-k(\d+)", name)

    if match:
        return int(match.group(1))

    return None


def load_cvrplib_instance(path: str | Path) -> CVRPInstance:
    """
    Lit une instance CVRPLIB au format .vrp avec vrplib.
    
    vrplib convertit généralement le dépôt en index 0.
    Exemple : si le fichier indique DEPOT_SECTION = 1,
    le dictionnaire Python aura depot = [0].
    """
    path = Path(path)
    raw = vrplib.read_instance(str(path))

    name = str(raw.get("name", path.stem))
    capacity = int(raw["capacity"])

    coords = np.asarray(raw["node_coord"], dtype=float)
    demands = np.asarray(raw["demand"], dtype=int)
    distance_matrix = np.asarray(raw["edge_weight"], dtype=float)

    depot_array = np.asarray(raw["depot"]).ravel()
    depot = int(depot_array[0])

    vehicle_count = raw.get("vehicles")

    if vehicle_count is not None:
        vehicle_count = int(vehicle_count)
    else:
        vehicle_count = parse_vehicle_count_from_name(name)

    return CVRPInstance(
        name=name,
        capacity=capacity,
        depot=depot,
        coords=coords,
        demands=demands,
        distance_matrix=distance_matrix,
        vehicle_count=vehicle_count,
    )