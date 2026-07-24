"""Comparative Evaluation Suite for POMDP-BKT Intelligent Tutoring System.

This module benchmarks the PyTorch Dueling Double DQN (D3QN) agent against
standard DQN and pedagogical Heuristic baselines on POMDPTutorEnv. It generates
statistical comparison summaries and saves graphical analysis charts to `artifacts/`.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pomdp_env import POMDPTutorEnv
from src.baselines import BaseTutor, RandomTutor, HeuristicTutor
from src.d3qn_agent import D3QNAgent, D3QNTutor, train_d3qn


class POMDPHeuristicTutor(BaseTutor):
    """Pedagogical Heuristic Tutor adapted for 5-action POMDPTutorEnv.

    Rules:
        - Checks rolling accuracy and current target skill belief.
        - If belief < 0.35: Presents Worked Example (Action 3).
        - If last answer was incorrect: Provides Scaffolding Hint (Action 4) or Easy Problem (Action 0).
        - If 2 consecutive correct answers: Advances difficulty (Easy -> Medium -> Hard).
    """

    def __init__(self) -> None:
        """Initializes POMDPHeuristicTutor."""
        self.current_difficulty_idx: int = 0
        self.consecutive_correct: int = 0

    def select_action(self, observation: np.ndarray) -> int:
        """Selects pedagogical action index in {0, 1, 2, 3, 4}.

        Args:
            observation: 8D state vector.

        Returns:
            int: Selected action index.
        """
        if len(observation) < 8:
            return 0

        mean_belief = float(observation[3])
        last_correct = float(observation[4])

        # Rule 1: If mean belief is low, present Worked Example
        if mean_belief < 0.35:
            return 3  # Worked Example

        # Rule 2: If recent answer incorrect, provide Scaffolding Hint
        if last_correct < 0.5:
            self.consecutive_correct = 0
            return 4  # Scaffolding Hint

        # Rule 3: Advance difficulty on consecutive correct answers
        self.consecutive_correct += 1
        if self.consecutive_correct >= 2:
            self.current_difficulty_idx = min(2, self.current_difficulty_idx + 1)
            self.consecutive_correct = 0

        return self.current_difficulty_idx

    def reset(self) -> None:
        """Resets internal state."""
        self.current_difficulty_idx = 0
        self.consecutive_correct = 0


def evaluate_pomdp_tutor(
    tutor: BaseTutor,
    env: POMDPTutorEnv,
    n_episodes: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates a tutor policy on POMDPTutorEnv over multiple episodes.

    Args:
        tutor: Policy instance implementing BaseTutor interface.
        env: POMDPTutorEnv environment instance.
        n_episodes: Number of simulation test episodes.
        seed: Random seed.

    Returns:
        Dict[str, Any]: Evaluation summary metrics and trajectory records.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    episode_rewards: List[float] = []
    final_mean_beliefs: List[float] = []
    episode_nlgs: List[float] = []
    episode_masteries: List[bool] = []
    episode_steps: List[int] = []

    belief_trajectories: List[List[float]] = []
    action_trajectories: List[List[int]] = []

    for ep in range(n_episodes):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)
        tutor.reset()

        ep_reward = 0.0
        step_count = 0
        ep_nlgs: List[float] = []
        terminated = False
        truncated = False

        b_traj: List[float] = [float(info["mean_belief"])]
        a_traj: List[int] = []

        while not (terminated or truncated):
            action = tutor.select_action(obs)
            a_traj.append(int(action))

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += float(reward)
            step_count += 1

            if "nlg" in step_info:
                ep_nlgs.append(float(step_info["nlg"]))

            b_traj.append(float(step_info["mean_belief"]))

        episode_rewards.append(ep_reward)
        final_mean_beliefs.append(float(b_traj[-1]))
        episode_nlgs.append(float(np.mean(ep_nlgs)) if ep_nlgs else 0.0)
        episode_masteries.append(bool(terminated))
        episode_steps.append(step_count)

        belief_trajectories.append(b_traj)
        action_trajectories.append(a_traj)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_nlg": float(np.mean(episode_nlgs)),
        "std_nlg": float(np.std(episode_nlgs)),
        "mean_final_belief": float(np.mean(final_mean_beliefs)),
        "std_final_belief": float(np.std(final_mean_beliefs)),
        "mastery_rate": float(np.mean(episode_masteries)),
        "mean_steps": float(np.mean(episode_steps)),
        "belief_trajectories": belief_trajectories,
        "action_trajectories": action_trajectories,
    }


def run_pomdp_benchmark(
    n_episodes: int = 50,
    seed: int = 42,
    model_path: str = "models/d3qn_tutor.pt",
    artifacts_dir: str = "artifacts",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs comparative evaluation comparing Random, Heuristic, and PyTorch D3QN tutors.

    Args:
        n_episodes: Evaluation episodes per policy.
        seed: Random seed.
        model_path: Path to trained D3QN checkpoint file.
        artifacts_dir: Directory path to save generated analysis charts.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Summary table and raw evaluation dictionary.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    env = POMDPTutorEnv(max_steps=30)

    # 1. Load or train D3QN agent
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' not found. Training D3QN agent...")
        train_d3qn(env, total_episodes=500, save_path=model_path, seed=seed)

    d3qn_agent = D3QNAgent(state_dim=8, action_dim=5, seed=seed)
    d3qn_agent.load(model_path)
    d3qn_tutor = D3QNTutor(d3qn_agent)

    random_tutor = RandomTutor(env.action_space, seed=seed)
    heuristic_tutor = POMDPHeuristicTutor()

    print(f"\n--- Running POMDP-BKT Benchmark ({n_episodes} Episodes, Seed={seed}) ---")

    rand_res = evaluate_pomdp_tutor(random_tutor, env, n_episodes=n_episodes, seed=seed)
    heur_res = evaluate_pomdp_tutor(heuristic_tutor, env, n_episodes=n_episodes, seed=seed)
    d3qn_res = evaluate_pomdp_tutor(d3qn_tutor, env, n_episodes=n_episodes, seed=seed)

    raw_results = {
        "Random Tutor": rand_res,
        "Heuristic Tutor": heur_res,
        "D3QN Agent (RL)": d3qn_res,
    }

    summary_df = pd.DataFrame({
        "Policy": ["Random Tutor", "Heuristic Tutor", "D3QN Agent (RL)"],
        "Mean Reward": [
            f"{rand_res['mean_reward']:.2f} ± {rand_res['std_reward']:.2f}",
            f"{heur_res['mean_reward']:.2f} ± {heur_res['std_reward']:.2f}",
            f"{d3qn_res['mean_reward']:.2f} ± {d3qn_res['std_reward']:.2f}",
        ],
        "Mean NLG": [
            f"{rand_res['mean_nlg']:.3f} ± {rand_res['std_nlg']:.3f}",
            f"{heur_res['mean_nlg']:.3f} ± {heur_res['std_nlg']:.3f}",
            f"{d3qn_res['mean_nlg']:.3f} ± {d3qn_res['std_nlg']:.3f}",
        ],
        "Final Skill Belief": [
            f"{rand_res['mean_final_belief']:.3f} ± {rand_res['std_final_belief']:.3f}",
            f"{heur_res['mean_final_belief']:.3f} ± {heur_res['std_final_belief']:.3f}",
            f"{d3qn_res['mean_final_belief']:.3f} ± {d3qn_res['std_final_belief']:.3f}",
        ],
        "Mastery Rate": [
            f"{rand_res['mastery_rate'] * 100:.1f}%",
            f"{heur_res['mastery_rate'] * 100:.1f}%",
            f"{d3qn_res['mastery_rate'] * 100:.1f}%",
        ],
        "Mean Steps": [
            f"{rand_res['mean_steps']:.1f}",
            f"{heur_res['mean_steps']:.1f}",
            f"{d3qn_res['mean_steps']:.1f}",
        ],
    })

    print("\n" + summary_df.to_string(index=False))

    # --- Generate Analysis Charts and Save to artifacts/ ---
    max_len = 31

    # 1. Belief Trajectory Line Chart
    plt.figure(figsize=(10, 5))
    for name, res in raw_results.items():
        padded = [
            t + [t[-1]] * (max_len - len(t)) if len(t) < max_len else t[:max_len]
            for t in res["belief_trajectories"]
        ]
        mean_b = np.mean(padded, axis=0)
        plt.plot(range(max_len), mean_b, label=name, linewidth=2.5)

    plt.title("Student Skill Belief Trajectory Progression (BKT Mean Belief)", fontsize=13, fontweight="bold")
    plt.xlabel("Step Count in Episode")
    plt.ylabel("Mean Skill Belief P(L_t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "pomdp_belief_trajectories.png"), dpi=200)
    plt.close()

    # 2. NLG and Reward Comparison Bar Chart
    fig, ax1 = plt.subplots(figsize=(8, 5))
    policies = list(raw_results.keys())
    rewards = [res["mean_reward"] for res in raw_results.values()]
    nlgs = [res["mean_nlg"] for res in raw_results.values()]

    x = np.arange(len(policies))
    width = 0.35

    ax1.bar(x - width / 2, rewards, width, label="Mean Reward", color="#3B82F6")
    ax1.set_ylabel("Mean Cumulative Reward", color="#3B82F6")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, nlgs, width, label="Mean NLG", color="#10B981")
    ax2.set_ylabel("Normalized Learning Gain (NLG)", color="#10B981")

    plt.title("Reward vs. Normalized Learning Gain (NLG) Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "pomdp_nlg_comparison.png"), dpi=200)
    plt.close()

    print(f"\nGraphical benchmark charts saved to '{artifacts_dir}/'.")
    return summary_df, raw_results


if __name__ == "__main__":
    run_pomdp_benchmark()
