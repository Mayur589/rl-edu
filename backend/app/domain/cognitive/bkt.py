"""Bayesian Knowledge Tracing (BKT) Engine for Multi-Skill POMDP Belief Tracking.

Maintains probabilistic posterior distributions over unobservable student skill masteries:
    b_t^(k) = P(L_t^(k) = 1)
Incorporates learning transitions, slip and guess noise parameters, scaffolding boosts,
and spaced memory retention decay.
"""

from typing import Dict, List, Optional, Union
import numpy as np

from app.core.config import settings
from app.core.exceptions import InvalidCognitiveStateException
from app.domain.cognitive.forgetting import RetentionDecayModel
from app.domain.curriculum.repository import KC_NAMES


class BKTParams:
    """Parameters governing Bayesian Knowledge Tracing updates for a skill component.

    Attributes:
        p_init: Prior probability of initial mastery P(L_0) in [0.0, 1.0].
        p_transit: Transition/learning probability P(T) in [0.0, 1.0].
        p_slip: Slip probability P(S) (answering wrong despite mastery) in [0.0, 1.0].
        p_guess: Guess probability P(G) (answering right despite non-mastery) in [0.0, 1.0].
        hint_boost: Additional learning transition boost when scaffolding/hint is provided.
    """

    def __init__(
        self,
        p_init: float = settings.BKT_P_INIT,
        p_transit: float = settings.BKT_P_TRANSIT,
        p_slip: float = settings.BKT_P_SLIP,
        p_guess: float = settings.BKT_P_GUESS,
        hint_boost: float = settings.BKT_HINT_BOOST,
    ) -> None:
        for name, val in [
            ("p_init", p_init),
            ("p_transit", p_transit),
            ("p_slip", p_slip),
            ("p_guess", p_guess),
            ("hint_boost", hint_boost),
        ]:
            if not (0.0 <= val <= 1.0):
                raise InvalidCognitiveStateException(
                    f"BKT parameter '{name}' must be bounded in [0.0, 1.0], got {val}"
                )

        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess
        self.hint_boost = hint_boost


class BKTEngine:
    """Multi-Skill Bayesian Knowledge Tracing Engine.

    Maintains a continuous belief vector b_t in [0, 1]^K across K Knowledge Components.
    """

    DEFAULT_SKILLS: List[str] = list(KC_NAMES)

    def __init__(
        self,
        skills: Optional[List[str]] = None,
        params: Optional[Dict[str, BKTParams]] = None,
        decay_model: Optional[RetentionDecayModel] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.skills = list(skills) if skills else list(self.DEFAULT_SKILLS)
        self.num_skills = len(self.skills)
        self.rng = np.random.default_rng(seed)
        self.decay_model = decay_model or RetentionDecayModel(decay_rate=settings.BKT_FORGETTING_RATE)

        # Initialize BKT parameters per KC
        self.params: Dict[str, BKTParams] = {}
        for skill in self.skills:
            self.params[skill] = params[skill] if (params and skill in params) else BKTParams()

        # Belief state vector b_t
        self.beliefs = np.zeros(self.num_skills, dtype=np.float32)
        # Tracking elapsed steps since last interaction per KC
        self.steps_since_touch: List[int] = [0] * self.num_skills
        # Historical belief trajectory for explainability / visualization
        self.belief_history: List[List[float]] = []

        self.reset()

    def reset(
        self, initial_beliefs: Optional[Union[List[float], np.ndarray]] = None
    ) -> np.ndarray:
        """Resets belief states to initial priors with stochastic variation."""
        if initial_beliefs is not None:
            arr = np.array(initial_beliefs, dtype=np.float32)
            if arr.shape != (self.num_skills,):
                raise InvalidCognitiveStateException(
                    f"initial_beliefs shape must be ({self.num_skills},), got {arr.shape}"
                )
            self.beliefs = np.clip(arr, 0.01, 0.99)
        else:
            for idx, skill in enumerate(self.skills):
                p_init = self.params[skill].p_init
                noise = float(self.rng.uniform(-0.04, 0.04))
                self.beliefs[idx] = float(np.clip(p_init + noise, 0.05, 0.95))

        self.steps_since_touch = [0] * self.num_skills
        self.belief_history = [self.beliefs.tolist()]
        return self.beliefs.copy()

    def update_belief(
        self,
        skill_idx: int,
        is_correct: bool,
        receives_scaffolding: bool = False,
    ) -> float:
        """Performs Bayesian posterior estimation and learning transition.

        Step 1: Posterior Likelihood (Bayes' Rule)
            P(L_t = 1 | O_t = 1) = b_t * (1 - p_slip) / [b_t * (1 - p_slip) + (1 - b_t) * p_guess]
            P(L_t = 1 | O_t = 0) = b_t * p_slip / [b_t * p_slip + (1 - b_t) * (1 - p_guess)]

        Step 2: Transition / Learning Update
            b_{t+1} = P(L_t = 1 | O_t) + (1 - P(L_t = 1 | O_t)) * P(T_effective)

        Args:
            skill_idx: Index of targeted Knowledge Component.
            is_correct: True if observed response was correct.
            receives_scaffolding: True if scaffolding hint/worked example was presented.

        Returns:
            float: Updated belief state for the targeted KC.
        """
        if not (0 <= skill_idx < self.num_skills):
            raise IndexError(f"skill_idx out of bounds: {skill_idx}")

        skill_name = self.skills[skill_idx]
        p = self.params[skill_name]
        b_t = float(self.beliefs[skill_idx])

        # 1. Posterior Likelihood Update
        if is_correct:
            num = b_t * (1.0 - p.p_slip)
            den = num + (1.0 - b_t) * p.p_guess
        else:
            num = b_t * p.p_slip
            den = num + (1.0 - b_t) * (1.0 - p.p_guess)

        den = max(1e-7, den)
        p_posterior = num / den

        # 2. Learning Transition
        p_transit_eff = p.p_transit + (p.hint_boost if receives_scaffolding else 0.0)
        p_transit_eff = min(1.0, p_transit_eff)

        b_next = p_posterior + (1.0 - p_posterior) * p_transit_eff
        b_next = float(np.clip(b_next, 0.01, 0.99))

        self.beliefs[skill_idx] = b_next

        # 3. Step age updates
        for idx in range(self.num_skills):
            if idx == skill_idx:
                self.steps_since_touch[idx] = 0
            else:
                self.steps_since_touch[idx] += 1
                # Apply passive forgetting decay to non-active skills
                decayed = self.decay_model.calculate_decayed_belief(
                    self.beliefs[idx], self.steps_since_touch[idx]
                )
                self.beliefs[idx] = decayed

        self.belief_history.append(self.beliefs.tolist())
        return b_next

    def get_beliefs(self) -> np.ndarray:
        """Returns copy of current belief vector."""
        return self.beliefs.copy()

    def get_mean_belief(self) -> float:
        """Calculates arithmetic mean belief across all KCs."""
        return float(np.mean(self.beliefs))

    def get_review_urgency(self) -> float:
        """Computes current normalized review urgency."""
        return self.decay_model.compute_review_urgency(self.beliefs, self.steps_since_touch)
