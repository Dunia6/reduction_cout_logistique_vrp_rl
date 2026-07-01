from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Tuple


@dataclass
class CVRPInstance:
    """
    Représente une instance CVRP lue depuis un fichier CVRPLIB.

    Attributs principaux :
    - name : nom de l'instance, ex: A-n32-k5
    - dimension : nombre total de noeuds, dépôt inclus
    - capacity : capacité maximale de chaque véhicule
    - vehicle_count : nombre de véhicules si disponible dans le nom ou le fichier
    - depot : identifiant du dépôt
    - coordinates : dictionnaire {noeud: (x, y)}
    - demands : dictionnaire {noeud: demande}
    - edge_weight_type : type de distance, souvent EUC_2D
    - reference_cost : coût de référence si fourni manuellement
    """

    name: str
    dimension: int
    capacity: int
    vehicle_count: int | None
    depot: int
    coordinates: Dict[int, Tuple[float, float]]
    demands: Dict[int, int]
    edge_weight_type: str = "EUC_2D"
    reference_cost: float | None = None

    @property
    def nodes(self) -> List[int]:
        return sorted(self.coordinates.keys())

    @property
    def customers(self) -> List[int]:
        return [node for node in self.nodes if node != self.depot]

    @property
    def customer_count(self) -> int:
        return len(self.customers)

    def distance(self, i: int, j: int) -> float:
        """
        Calcule la distance entre deux noeuds.

        Pour les instances CVRPLIB de type EUC_2D, on utilise généralement
        la distance euclidienne arrondie à l'entier le plus proche.
        """
        if i == j:
            return 0.0

        xi, yi = self.coordinates[i]
        xj, yj = self.coordinates[j]

        dist = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

        if self.edge_weight_type.upper() == "EUC_2D":
            return float(round(dist))

        return dist

    def route_cost(self, route: List[int]) -> float:
        """
        Calcule le coût d'une route.
        Exemple de route valide : [1, 5, 8, 12, 1]
        """
        if len(route) < 2:
            return 0.0

        return sum(
            self.distance(route[idx], route[idx + 1])
            for idx in range(len(route) - 1)
        )

    def solution_cost(self, routes: List[List[int]]) -> float:
        """
        Calcule le coût total d'une solution CVRP.
        """
        return sum(self.route_cost(route) for route in routes)

    def route_demand(self, route: List[int]) -> int:
        """
        Calcule la demande totale servie par une route.
        Le dépôt est ignoré car sa demande est normalement égale à zéro.
        """
        return sum(
            self.demands.get(node, 0)
            for node in route
            if node != self.depot
        )