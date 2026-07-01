from pprint import pprint
from time import perf_counter
import random

from src.cvrp.data.parser import load_cvrp_instance
from src.cvrp.rl.env import CVRPEnv
from src.cvrp.rl.q_learning_agent import QLearningAgent
from src.cvrp.core.metrics import build_final_result_row


def train_one_episode(
    env: CVRPEnv,
    agent: QLearningAgent,
    max_steps: int = 500,
) -> dict:
    """
    Entraîne l'agent pendant un épisode.
    """

    env.reset()

    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        state = env.get_q_learning_state()
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            break

        action = agent.select_action(
            state=state,
            valid_actions=valid_actions,
            greedy=False,
        )

        _, reward, done, info = env.step(action)

        next_state = env.get_q_learning_state()
        next_valid_actions = env.get_valid_actions() if not done else []

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_valid_actions=next_valid_actions,
            done=done,
        )

        total_reward += reward
        steps += 1

    agent.decay_epsilon()

    routes = env.get_solution()
    validation = env.validate_current_solution()

    return {
        "reward": total_reward,
        "cost": validation.cost,
        "gap_percent": validation.gap_percent,
        "is_valid": validation.is_valid,
        "routes": len(routes),
        "missing_customers_count": len(validation.missing_customers),
        "steps": steps,
        "epsilon": agent.epsilon,
        "q_table_size": agent.q_table_size(),
    }


def evaluate_greedy_policy(
    env: CVRPEnv,
    agent: QLearningAgent,
    max_steps: int = 500,
) -> tuple[list[list[int]], dict]:
    """
    Évalue l'agent en mode greedy, sans exploration.
    """

    env.reset()

    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        state = env.get_q_learning_state()
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            break

        action = agent.select_action(
            state=state,
            valid_actions=valid_actions,
            greedy=True,
        )

        _, reward, done, info = env.step(action)

        total_reward += reward
        steps += 1

    routes = env.get_solution()
    validation = env.validate_current_solution()

    evaluation = {
        "reward": total_reward,
        "cost": validation.cost,
        "gap_percent": validation.gap_percent,
        "is_valid": validation.is_valid,
        "routes": len(routes),
        "missing_customers_count": len(validation.missing_customers),
        "steps": steps,
    }

    return routes, evaluation


def main():
    seed = 42
    random.seed(seed)

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

    agent = QLearningAgent(
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=seed,
    )

    episodes = 50

    print("Instance :", instance.name)
    print("Épisodes de test :", episodes)
    print("Seed :", seed)

    training_metrics = []

    start_train = perf_counter()

    for episode in range(1, episodes + 1):
        metrics = train_one_episode(env, agent)
        metrics["episode"] = episode
        metrics["seed"] = seed
        training_metrics.append(metrics)

        if episode == 1 or episode % 10 == 0:
            print(
                f"Episode {episode:03d} | "
                f"reward={metrics['reward']:.2f} | "
                f"cost={metrics['cost']} | "
                f"valid={metrics['is_valid']} | "
                f"epsilon={metrics['epsilon']:.4f} | "
                f"q_states={metrics['q_table_size']}"
            )

    train_time = perf_counter() - start_train

    start_eval = perf_counter()
    routes, evaluation = evaluate_greedy_policy(env, agent)
    inference_time = perf_counter() - start_eval

    print("\n--- Évaluation greedy après entraînement ---")
    pprint(evaluation)

    print("\nRoutes obtenues :")
    for idx, route in enumerate(routes, start=1):
        print(f"Route {idx} :", route)

    row = build_final_result_row(
        instance=instance,
        method="Q-learning RL",
        category="rl_baseline",
        routes=routes,
        seed=seed,
        train_time_sec=train_time,
        inference_time_sec=inference_time,
        postprocess_time_sec=0.0,
        episodes=episodes,
        notes=(
            "Baseline RL tabulaire. "
            "Test rapide ; le vrai entraînement sera fait dans le notebook."
        ),
    )

    print("\nLigne standardisée finale :")
    pprint(row)

    output_path = "results/q_learning/q_learning_agent_test.json"
    agent.save(output_path)
    print("\nAgent sauvegardé :", output_path)


if __name__ == "__main__":
    main()