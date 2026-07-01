from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.paths import INSTANCES_DIR


def main():
    instance = load_cvrp_instance(
        INSTANCES_DIR / "A-n32-k5.vrp",
        reference_cost=784,
    )

    print("Instance :", instance.name)
    print("Dimension :", instance.dimension)
    print("Clients :", instance.customer_count)
    print("Dépôt :", instance.depot)
    print("Véhicules :", instance.vehicle_count)
    print("Capacité :", instance.capacity)
    print("Référence :", instance.reference_cost)
    print("Type de distance :", instance.edge_weight_type)

    print("\nPremiers clients :", instance.customers[:5])
    print("Demande totale :", sum(instance.demands[i] for i in instance.customers))

    d = instance.distance(instance.depot, instance.customers[0])
    print(f"Distance dépôt → client {instance.customers[0]} :", d)


if __name__ == "__main__":
    main()
