from pprint import pprint
from time import perf_counter

from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.rl.env import CVRPEnv
from src.cvrp.core.metrics import build_final_result_row


def choose_greedy_valid_action(env: CVRPEnv) -> int:
    """
    Petite politique de test :
    - choisir le client valide le plus proche ;
    - sinon retourner au dépôt.
    """

    instance = env.instance
    depot = instance.depot

    valid_actions = env.get_valid_actions()

    if not valid_actions:
        raise RuntimeError("Aucune action valide disponible.")

    customer_actions = [action for action in valid_actions if action != depot]

    if customer_actions:
        return min(
            customer_actions,
            key=lambda customer: instance.distance(env.current_node, customer),
        )

    return depot


def main():
    instance = load_cvrp_instance(
        "data/instances/A-n32-k5.vrp",
        reference_cost=784,
    )

    env = CVRPEnv(
        instance=instance,
        max_routes=instance.vehicle_count,
        invalid_action_penalty=-1000,
        incomplete_solution_penalty=-5000,
        completion_bonus=100,
        distance_reward_scale=1.0,
    )

    start = perf_counter()

    observation = env.reset()
    done = False

    while not done:
        action = choose_greedy_valid_action(env)
        observation, reward, done, info = env.step(action)

    inference_time = perf_counter() - start

    routes = env.get_solution()
    validation = env.validate_current_solution()

    print("Instance :", instance.name)
    print("Solution valide :", validation.is_valid)
    print("Coût :", validation.cost)
    print("Gap (%) :", validation.gap_percent)
    print("Routes :", len(routes))
    print("Charges :", validation.route_loads)
    print("Reward total :", env.total_reward)
    print("Distance totale :", env.total_distance)
    print("Étapes :", env.steps)

    print("\nRoutes :")
    for idx, route in enumerate(routes, start=1):
        print(f"Route {idx} :", route)

    if validation.errors:
        print("\nErreurs :")
        for error in validation.errors:
            print("-", error)

    print("\nÉtat Q-learning final :")
    print(env.get_q_learning_state())

    print("\nNombre de features par noeud :")
    node_features = env.get_node_features()
    print("Noeuds :", len(node_features))
    print("Features par noeud :", len(node_features[0]))

    row = build_final_result_row(
        instance=instance,
        method="Greedy policy through RL Env",
        category="rl_env_test",
        routes=routes,
        seed=None,
        train_time_sec=0.0,
        inference_time_sec=inference_time,
        postprocess_time_sec=0.0,
        episodes=1,
        notes="Test de l'environnement RL avec une politique gloutonne.",
    )

    print("\nLigne standardisée :")
    pprint(row)


if __name__ == "__main__":
    main()