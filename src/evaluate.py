"""Evaluation Module for Benchmark and RL Tutor Policies.

This module provides standardized evaluation procedures for measuring tutor performance
across multiple simulated episodes, recording cumulative rewards, learning gains,
accuracy, mastery rates, and trajectory data.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.environment import SimulatedStudentEnv
from src.baselines import BaseTutor, RandomTutor, HeuristicTutor


def evaluate_tutor(
    tutor: BaseTutor,
    env: SimulatedStudentEnv,
    n_episodes: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates a tutor policy over multiple environment episodes.

    Args:
        tutor: Policy implementing `BaseTutor` interface (or wrapper with `select_action`).
        env: `SimulatedStudentEnv` environment instance.
        n_episodes: Number of simulation episodes to run. Must be > 0.
        seed: Random seed for deterministic environment reset sequence.

    Returns:
        Dict[str, Any]: Dictionary containing aggregated metrics and trajectory histories:
            - "mean_reward": float
            - "std_reward": float
            - "mean_final_knowledge": float
            - "std_final_knowledge": float
            - "mean_accuracy": float
            - "mastery_rate": float
            - "mean_steps": float
            - "knowledge_trajectories": List[List[float]]
            - "difficulty_trajectories": List[List[int]]

    Raises:
        ValueError: If n_episodes <= 0.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    episode_rewards: List[float] = []
    final_knowledges: List[float] = []
    episode_accuracies: List[float] = []
    episode_masteries: List[bool] = []
    episode_steps: List[int] = []

    knowledge_trajectories: List[List[float]] = []
    difficulty_trajectories: List[List[int]] = []

    for episode in range(n_episodes):
        ep_seed = seed + episode
        obs, info = env.reset(seed=ep_seed)
        tutor.reset()

        ep_reward = 0.0
        correct_count = 0
        step_count = 0
        terminated = False
        truncated = False

        k_traj: List[float] = [float(info["student_knowledge"])]
        d_traj: List[int] = []

        while not (terminated or truncated):
            action = tutor.select_action(obs)
            d_traj.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            step_count += 1

            if info.get("is_correct", False):
                correct_count += 1

            k_traj.append(float(info["student_knowledge"]))

        episode_rewards.append(ep_reward)
        final_knowledges.append(float(info["student_knowledge"]))
        episode_accuracies.append(correct_count / max(1, step_count))
        episode_masteries.append(bool(terminated))
        episode_steps.append(step_count)

        knowledge_trajectories.append(k_traj)
        difficulty_trajectories.append(d_traj)

    results: Dict[str, Any] = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_final_knowledge": float(np.mean(final_knowledges)),
        "std_final_knowledge": float(np.std(final_knowledges)),
        "mean_accuracy": float(np.mean(episode_accuracies)),
        "mastery_rate": float(np.mean(episode_masteries)),
        "mean_steps": float(np.mean(episode_steps)),
        "knowledge_trajectories": knowledge_trajectories,
        "difficulty_trajectories": difficulty_trajectories,
    }
    return results


def run_baseline_comparison(
    n_episodes: int = 50, seed: int = 42
) -> pd.DataFrame:
    """Runs evaluation benchmark comparing Random and Heuristic baselines.

    Args:
        n_episodes: Number of test episodes per baseline.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Summary comparison table.
    """
    env = SimulatedStudentEnv(max_steps=30)

    random_tutor = RandomTutor(env.action_space, seed=seed)
    heuristic_tutor = HeuristicTutor(initial_difficulty=1)

    print(f"\n--- Running Baseline Evaluation ({n_episodes} Episodes, Seed={seed}) ---")

    rand_results = evaluate_tutor(random_tutor, env, n_episodes=n_episodes, seed=seed)
    heur_results = evaluate_tutor(heuristic_tutor, env, n_episodes=n_episodes, seed=seed)

    summary_data = {
        "Policy": ["Random Tutor", "Heuristic Tutor"],
        "Mean Reward": [
            f"{rand_results['mean_reward']:.2f} ± {rand_results['std_reward']:.2f}",
            f"{heur_results['mean_reward']:.2f} ± {heur_results['std_reward']:.2f}",
        ],
        "Final Knowledge": [
            f"{rand_results['mean_final_knowledge']:.3f} ± {rand_results['std_final_knowledge']:.3f}",
            f"{heur_results['mean_final_knowledge']:.3f} ± {heur_results['std_final_knowledge']:.3f}",
        ],
        "Accuracy": [
            f"{rand_results['mean_accuracy'] * 100:.1f}%",
            f"{heur_results['mean_accuracy'] * 100:.1f}%",
        ],
        "Mastery Rate": [
            f"{rand_results['mastery_rate'] * 100:.1f}%",
            f"{heur_results['mastery_rate'] * 100:.1f}%",
        ],
        "Mean Steps": [
            f"{rand_results['mean_steps']:.1f}",
            f"{heur_results['mean_steps']:.1f}",
        ],
    }

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_baseline_comparison()
