"""Cognitive Modeling & Memory Retention Decay.

Implements Ebbinghaus-inspired retention decay to model spaced repetition dynamics.
"""

import math
from typing import List
import numpy as np


class RetentionDecayModel:
    """Calculates time-dependent forgetting and review urgency for skills.

    Attributes:
        decay_rate: Lambda parameter controlling forgetting slope.
        retention_floor: Minimum baseline retention asymptote (e.g. 0.10).
    """

    def __init__(self, decay_rate: float = 0.02, retention_floor: float = 0.10) -> None:
        self.decay_rate = decay_rate
        self.retention_floor = retention_floor

    def calculate_decayed_belief(
        self,
        current_belief: float,
        steps_since_last_touch: int,
    ) -> float:
        """Calculates decayed belief after unvisited elapsed time steps.

        b(t) = b_floor + (b_current - b_floor) * exp(-decay_rate * delta_t)

        Args:
            current_belief: Knowledge belief in [0.0, 1.0].
            steps_since_last_touch: Number of steps since KC was practiced.

        Returns:
            float: Decayed knowledge belief.
        """
        if steps_since_last_touch <= 0:
            return current_belief

        decay_factor = math.exp(-self.decay_rate * steps_since_last_touch)
        decayed = self.retention_floor + (current_belief - self.retention_floor) * decay_factor
        return float(np.clip(decayed, self.retention_floor, 1.0))

    def compute_review_urgency(
        self,
        beliefs: np.ndarray,
        steps_since_touch: List[int],
    ) -> float:
        """Computes a normalized scalar review urgency index in [0, 1].

        High urgency occurs when previously mastered skills haven't been reviewed
        and are at risk of cognitive decay.

        Args:
            beliefs: Array of current skill beliefs.
            steps_since_touch: Steps elapsed since each skill was touched.

        Returns:
            float: Normalized review urgency index in [0.0, 1.0].
        """
        urgencies: List[float] = []
        for b, steps in zip(beliefs, steps_since_touch):
            # If skill had substantial belief (> 0.5) and many steps passed, urgency spikes
            if b > 0.4:
                urgency = (1.0 - math.exp(-0.08 * steps)) * b
                urgencies.append(urgency)
            else:
                urgencies.append(0.0)

        return float(np.clip(max(urgencies) if urgencies else 0.0, 0.0, 1.0))
