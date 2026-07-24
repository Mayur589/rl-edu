"""Reinforcement Learning (DQN) Agent for Intelligent Tutoring System.

This module provides functions to instantiate, train, save, load, and evaluate
a Deep Q-Network (DQN) agent using Stable-Baselines3.
"""

import os
import random
from typing import Optional, Union
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from src.environment import SimulatedStudentEnv
from src.baselines import BaseTutor


def set_random_seeds(seed: int = 42) -> None:
    """Sets explicit random seeds across numpy, torch, random, and python for reproducibility.

    Args:
        seed: Integer random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DQNTutor(BaseTutor):
    """Wrapper class adapting a trained Stable-Baselines3 DQN model to the `BaseTutor` interface.

    Attributes:
        model (DQN): Trained Stable-Baselines3 DQN policy.
        deterministic (bool): Whether to select greedy actions during evaluation.
    """

    def __init__(self, model: DQN, deterministic: bool = True) -> None:
        """Initializes DQNTutor.

        Args:
            model: Instantiated or loaded DQN model.
            deterministic: If True, uses deterministic (greedy) action selection.
        """
        self.model: DQN = model
        self.deterministic: bool = deterministic

    def select_action(self, observation: np.ndarray) -> int:
        """Selects action using trained DQN policy network.

        Args:
            observation: Environment state vector of shape (7,).

        Returns:
            int: Action index in {0, 1, 2}.
        """
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(action)

    def reset(self) -> None:
        """Resets internal state (no-op for static DQN policy)."""
        pass


def train_dqn_agent(
    env: gym.Env,
    total_timesteps: int = 30000,
    seed: int = 42,
    model_save_path: str = "models/dqn_tutor.zip",
) -> DQN:
    """Trains a DQN agent on the given student environment and saves model weights.

    Args:
        env: Gymnasium environment instance.
        total_timesteps: Total environment interaction steps for RL training.
        seed: Random seed for deterministic model initialization and environment.
        model_save_path: Destination filepath to save trained model (.zip format).

    Returns:
        DQN: Trained Stable-Baselines3 DQN model.

    Raises:
        ValueError: If total_timesteps <= 0.
    """
    if total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")

    set_random_seeds(seed)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=250,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        seed=seed,
        verbose=0,
    )

    print(f"Training DQN agent for {total_timesteps} timesteps (Seed={seed})...")
    model.learn(total_timesteps=total_timesteps)

    # Ensure parent output directory exists
    dir_name = os.path.dirname(model_save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    model.save(model_save_path)
    print(f"Model saved successfully to '{model_save_path}'.")
    return model


def load_dqn_agent(
    model_path: str = "models/dqn_tutor.zip",
    env: Optional[gym.Env] = None,
) -> DQN:
    """Loads a pre-trained DQN model from disk.

    Args:
        model_path: Path to model zip file.
        env: Optional environment instance to attach to model.

    Returns:
        DQN: Loaded Stable-Baselines3 model.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: '{model_path}'")

    model = DQN.load(model_path, env=env)
    return model


if __name__ == "__main__":
    env = SimulatedStudentEnv(max_steps=30)
    trained_model = train_dqn_agent(env, total_timesteps=30000, seed=42)
