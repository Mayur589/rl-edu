"""Experience Replay Buffer for Off-Policy Reinforcement Learning.

Stores and samples transition tuples (s, a, r, s', done) in pure NumPy.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class ReplayBuffer:
    """Fixed-capacity ring buffer for storing experience tuples."""

    def __init__(
        self,
        capacity: int = 10000,
        state_dim: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a new experience tuple to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draws a uniform random mini-batch of experiences."""
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")

        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size
