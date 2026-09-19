"""Pedagogical Baseline Policies for Comparative Benchmarking.

Implements standard benchmark policies:
1. RandomPolicy: Uniform random action selection.
2. LinearCurriculumPolicy: Static sequence (Easy -> Medium -> Hard) with no guidance adaptation.
3. RuleBasedHeuristicPolicy: Rule-based teacher heuristics balancing difficulty, hints, and friction.
4. LeitnerSpacedRepetitionPolicy: Classical flashcard spaced repetition prioritization.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePedagogicalPolicy(ABC):
    """Abstract base class for pedagogical tutoring policies."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Selects action given the 10-dimensional state vector."""
        pass


class RandomPolicy(BasePedagogicalPolicy):
    """Selects actions uniformly at random."""

    def __init__(self, action_dim: int = 6, seed: int = 42) -> None:
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray) -> int:
        return int(self.rng.integers(0, self.action_dim))


class LinearCurriculumPolicy(BasePedagogicalPolicy):
    """Fixed deterministic curriculum ignoring individual friction and guidance needs."""

    def __init__(self) -> None:
        pass

    def select_action(self, state: np.ndarray) -> int:
        # Steps progression: first 30% Easy, next 40% Medium, final 30% Hard
        step_ratio = state[7] if len(state) > 7 else 0.0
        if step_ratio < 0.30:
            return 0  # Easy
        elif step_ratio < 0.70:
            return 1  # Medium
        else:
            return 2  # Hard


class RuleBasedHeuristicPolicy(BasePedagogicalPolicy):
    """Expert-designed pedagogical heuristic policy.

    Adapts difficulty based on rolling accuracy, provides scaffolding when friction
    is high or student struggles, and schedules review when urgency spikes.
    """

    def __init__(self) -> None:
        pass

    def select_action(self, state: np.ndarray) -> int:
        mean_b = state[4]
        rolling_acc = state[5]
        friction = state[6]
        urgency = state[9]

        # 1. Spaced review trigger
        if urgency > 0.65:
            return 5  # Spaced Review

        # 2. Guidance adaptation trigger on high struggle / friction
        if friction > 0.60 or rolling_acc < 0.35:
            return 3  # Scaffolding Hint

        # 3. Content adaptation based on mastery
        if mean_b > 0.75 and rolling_acc >= 0.70:
            return 2  # Hard Problem
        elif mean_b > 0.40 and rolling_acc >= 0.50:
            return 1  # Medium Problem
        else:
            return 0  # Easy Problem


class LeitnerSpacedRepetitionPolicy(BasePedagogicalPolicy):
    """Prioritizes spaced repetition intervals and review whenever urgency is detectable."""

    def __init__(self) -> None:
        pass

    def select_action(self, state: np.ndarray) -> int:
        rolling_acc = state[5]
        urgency = state[9]

        if urgency > 0.40:
            return 5  # Spaced Review
        elif rolling_acc < 0.40:
            return 4  # Worked Example on error
        else:
            return 1  # Standard Medium practice
