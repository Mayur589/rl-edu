"""Custom PyTorch Dueling Double Deep Q-Network (D3QN) for POMDP Tutor.

This module implements:
1. `DuelingDQN`: Neural network decomposing Q-values into State Value V(s) and Advantage A(s, a).
2. `ReplayBuffer`: Experience replay memory buffer for off-policy RL sampling.
3. `D3QNAgent`: Double DQN learning agent with target network synchronization.
4. `D3QNTutor`: Adapter wrapper for evaluation and explainability.
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.pomdp_env import POMDPTutorEnv
from src.baselines import BaseTutor


def set_d3qn_seeds(seed: int = 42) -> None:
    """Sets explicit random seeds for PyTorch, NumPy, Python, and Cuda.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture.

    Decomposes state-action values Q(s, a) into a scalar State Value V(s) and an
    Action Advantage vector A(s, a):
        Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum_a' A(s, a'))
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 5, hidden_dim: int = 128) -> None:
        """Initializes DuelingDQN network architecture.

        Args:
            state_dim: Dimension of input state vector.
            action_dim: Number of discrete actions.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        # Shared feature extractor
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream V(s) -> scalar (batch, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream A(s, a) -> vector (batch, action_dim)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Computes Q-values Q(s, a) for all actions.

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Q-values tensor of shape (batch_size, action_dim).
        """
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Dueling aggregation formula
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

    def get_decomposed(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns decomposed V(s), A(s, a), and Q(s, a) for explainability visualization.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (Value, Advantage, Q-values).
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return values, advantages, q_values


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transition tuples."""

    def __init__(self, capacity: int = 50000, seed: Optional[int] = None) -> None:
        """Initializes ReplayBuffer.

        Args:
            capacity: Maximum number of transition tuples to hold.
            seed: Random seed for sampling.
        """
        self.capacity: int = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position: int = 0
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a transition tuple to the buffer.

        Args:
            state: Current state vector.
            action: Executed action index.
            reward: Scalar reward received.
            next_state: Next state vector.
            done: Termination flag.
        """
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            device: PyTorch device ('cpu' or 'cuda').

        Returns:
            Tuple of PyTorch tensors (states, actions, rewards, next_states, dones).
        """
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


class D3QNAgent:
    """Dueling Double DQN Agent with target network updates and Double Q-learning logic."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 5,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        """Initializes D3QNAgent.

        Args:
            state_dim: Dimension of state vector.
            action_dim: Number of actions.
            learning_rate: Optimizer learning rate.
            gamma: Discount factor gamma.
            buffer_capacity: Maximum buffer size.
            batch_size: Training batch size.
            seed: Random seed.
            device: Computing device ('cpu' or 'cuda').
        """
        set_d3qn_seeds(seed)
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.gamma: float = gamma
        self.batch_size: int = batch_size

        if device is None:
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Online network & Target network
        self.online_net: DuelingDQN = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net: DuelingDQN = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer: optim.Optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.memory: ReplayBuffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.loss_fn: nn.Module = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Selects action using epsilon-greedy policy.

        Args:
            state: Observation vector array.
            epsilon: Exploration probability in [0.0, 1.0].

        Returns:
            int: Selected action index.
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def train_step(self) -> Optional[float]:
        """Performs a single Double DQN optimization step over a sampled batch.

        Returns:
            Optional[float]: Loss value if step was performed, else None.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)

        self.online_net.train()

        # Current Q-values Q(s, a; theta_online)
        q_eval = self.online_net(states).gather(1, actions)

        # --- Double DQN Target Calculation ---
        # 1. Best next action selected by ONLINE network: a* = argmax_a Q(s', a; theta_online)
        with torch.no_grad():
            next_online_q = self.online_net(next_states)
            best_next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)

            # 2. Next Q-value evaluated by TARGET network: Q(s', a*; theta_target)
            next_target_q = self.target_net(next_states).gather(1, best_next_actions)

            # 3. Target Y = r + gamma * (1 - done) * Q_target(s', a*)
            q_target = rewards + (1.0 - dones) * self.gamma * next_target_q

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copies online network parameters to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, filepath: str) -> None:
        """Saves model weights to disk.

        Args:
            filepath: Destination file path.
        """
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )
        print(f"D3QN agent model saved successfully to '{filepath}'.")

    def load(self, filepath: str) -> None:
        """Loads model weights from disk.

        Args:
            filepath: Path to model checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"D3QN model checkpoint not found at: '{filepath}'")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.online_net.eval()
        self.target_net.eval()
        print(f"D3QN agent model loaded successfully from '{filepath}'.")


class D3QNTutor(BaseTutor):
    """Adapter class wrapping `D3QNAgent` for standardized evaluation and UI explainability."""

    def __init__(self, agent: D3QNAgent) -> None:
        """Initializes D3QNTutor.

        Args:
            agent: Instantiated or loaded D3QNAgent.
        """
        self.agent: D3QNAgent = agent

    def select_action(self, observation: np.ndarray) -> int:
        """Selects greedy action given observation.

        Args:
            observation: State observation vector array.

        Returns:
            int: Action index in {0, 1, 2, 3, 4}.
        """
        return self.agent.select_action(observation, epsilon=0.0)

    def predict_explainability(
        self, observation: np.ndarray
    ) -> Tuple[float, List[float], List[float]]:
        """Decomposes observation into State Value V(s), Advantages A(s, a), and Q-values Q(s, a).

        Args:
            observation: 8D state vector.

        Returns:
            Tuple containing:
                - float: Scalar State Value V(s).
                - List[float]: Advantage values per action A(s, a).
                - List[float]: Q-values per action Q(s, a).
        """
        state_t = torch.tensor(observation, dtype=torch.float32, device=self.agent.device)
        self.agent.online_net.eval()
        with torch.no_grad():
            v_t, a_t, q_t = self.agent.online_net.get_decomposed(state_t)

        v_val = float(v_t.squeeze().item())
        adv_list = a_t.squeeze().cpu().numpy().tolist()
        q_list = q_t.squeeze().cpu().numpy().tolist()

        return v_val, adv_list, q_list

    def reset(self) -> None:
        """Resets internal state (no-op)."""
        pass


def train_d3qn(
    env: POMDPTutorEnv,
    total_episodes: int = 500,
    max_steps_per_ep: int = 30,
    save_path: str = "models/d3qn_tutor.pt",
    seed: int = 42,
) -> D3QNAgent:
    """Trains D3QNAgent on POMDPTutorEnv and saves checkpoint.

    Args:
        env: POMDPTutorEnv environment instance.
        total_episodes: Number of episodes to train.
        max_steps_per_ep: Maximum steps per episode.
        save_path: Filepath destination for model checkpoint.
        seed: Random seed.

    Returns:
        D3QNAgent: Trained agent instance.
    """
    set_d3qn_seeds(seed)
    agent = D3QNAgent(state_dim=8, action_dim=5, seed=seed)

    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.992

    epsilon = eps_start
    target_update_freq = 10  # Update target network every 10 episodes

    print(f"\n--- Training PyTorch D3QN Agent ({total_episodes} Episodes, Seed={seed}) ---")

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0

        for step in range(max_steps_per_ep):
            action = agent.select_action(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(obs, action, reward, next_obs, done)
            agent.train_step()

            ep_reward += reward
            obs = next_obs

            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)

        if ep % target_update_freq == 0:
            agent.update_target_network()

        if ep % 100 == 0 or ep == total_episodes:
            print(f"Episode {ep:3d}/{total_episodes} | Ep Reward: {ep_reward:6.2f} | Epsilon: {epsilon:.3f}")

    agent.save(save_path)
    return agent


if __name__ == "__main__":
    env = POMDPTutorEnv(max_steps=30)
    train_d3qn(env, total_episodes=500, save_path="models/d3qn_tutor.pt", seed=42)
