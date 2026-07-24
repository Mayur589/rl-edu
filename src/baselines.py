"""Baseline Policy Tutors for Intelligent Tutoring System.

This module defines `RandomTutor` and `HeuristicTutor` baseline policies used
to benchmark Reinforcement Learning agents.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from gymnasium import spaces


class BaseTutor(ABC):
    """Abstract Base Class for all Tutor Policies."""

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """Selects a discrete action (difficulty level) given the current observation.

        Args:
            observation: State observation array from environment.

        Returns:
            int: Action index in {0, 1, 2}.
        """
        pass

    def reset(self) -> None:
        """Resets any internal state of the tutor between episodes."""
        pass


class RandomTutor(BaseTutor):
    """Random Tutor baseline that selects actions uniformly at random.

    Attributes:
        action_space (spaces.Discrete): The discrete action space.
        rng (np.random.Generator): Random number generator for deterministic sampling.
    """

    def __init__(
        self, action_space: spaces.Discrete, seed: Optional[int] = None
    ) -> None:
        """Initializes RandomTutor.

        Args:
            action_space: Discrete action space (e.g. spaces.Discrete(3)).
            seed: Seed for random number generator.

        Raises:
            ValueError: If action_space is not Discrete.
        """
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(f"RandomTutor requires Discrete action space, got {type(action_space)}")
        self.action_space: spaces.Discrete = action_space
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """Uniformly samples an action from the action space.

        Args:
            observation: Current environment observation (unused by random policy).

        Returns:
            int: Random action index in {0, 1, ..., n-1}.
        """
        return int(self.rng.choice(self.action_space.n))

    def reset(self) -> None:
        """Resets internal state (no-op for random policy)."""
        pass


class HeuristicTutor(BaseTutor):
    """Heuristic Rule-Based Tutor following pedagogical ZPD adaptation logic.

    Rules:
        - Starts at Medium difficulty (action 1).
        - Increases difficulty if student completes 2 consecutive correct answers.
        - Decreases difficulty if student gets 1 incorrect answer.

    Attributes:
        current_difficulty (int): Current action index (0: Easy, 1: Medium, 2: Hard).
        consecutive_correct (int): Tracked count of consecutive correct answers.
    """

    def __init__(self, initial_difficulty: int = 1) -> None:
        """Initializes HeuristicTutor.

        Args:
            initial_difficulty: Initial action index in {0, 1, 2}. Defaults to 1 (Medium).

        Raises:
            ValueError: If initial_difficulty is not in {0, 1, 2}.
        """
        if initial_difficulty not in (0, 1, 2):
            raise ValueError(f"initial_difficulty must be in {{0, 1, 2}}, got {initial_difficulty}")
        self.initial_difficulty: int = initial_difficulty
        self.current_difficulty: int = initial_difficulty
        self.consecutive_correct: int = 0

    def select_action(self, observation: np.ndarray) -> int:
        """Selects action based on pedagogical heuristics and state history.

        Args:
            observation: 7-dimensional environment observation array.
                observation[2] represents last_correctness (1.0 for correct, 0.0 for incorrect).

        Returns:
            int: Selected action index in {0, 1, 2}.
        """
        if len(observation) < 3:
            return self.current_difficulty

        # Extract last correctness from state observation
        last_correct = observation[2]

        if last_correct >= 0.5:
            self.consecutive_correct += 1
            # Increase difficulty if student scores 2 consecutive correct answers
            if self.consecutive_correct >= 2:
                self.current_difficulty = min(2, self.current_difficulty + 1)
                self.consecutive_correct = 0
        else:
            # Decrease difficulty on wrong answer
            self.consecutive_correct = 0
            self.current_difficulty = max(0, self.current_difficulty - 1)

        return self.current_difficulty

    def reset(self) -> None:
        """Resets internal tracking variables at episode start."""
        self.current_difficulty = self.initial_difficulty
        self.consecutive_correct = 0
