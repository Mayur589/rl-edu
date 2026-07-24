"""Bayesian Knowledge Tracing (BKT) Engine for POMDP Belief Tracking.

This module implements a multi-skill Bayesian Knowledge Tracing (BKT) engine that
maintains and updates probabilistic belief states over unobservable student skill masteries.
"""

from typing import Dict, List, Optional, Union
import numpy as np


class BKTParams:
    """Parameters governing Bayesian Knowledge Tracing updates for a skill component.

    Attributes:
        p_init (float): Prior probability of initial mastery P(L_0) in [0.0, 1.0].
        p_transit (float): Probability of learning/transition P(T) in [0.0, 1.0].
        p_slip (float): Probability of slip P(S) (incorrect despite mastery) in [0.0, 1.0].
        p_guess (float): Probability of guess P(G) (correct despite non-mastery) in [0.0, 1.0].
        hint_boost (float): Additional learning transition gain when scaffolding/hint is provided.
    """

    def __init__(
        self,
        p_init: float = 0.20,
        p_transit: float = 0.15,
        p_slip: float = 0.10,
        p_guess: float = 0.25,
        hint_boost: float = 0.10,
    ) -> None:
        """Initializes BKT parameters with input validation.

        Raises:
            ValueError: If any parameter is outside valid probability bounds [0.0, 1.0].
        """
        for param_name, val in [
            ("p_init", p_init),
            ("p_transit", p_transit),
            ("p_slip", p_slip),
            ("p_guess", p_guess),
            ("hint_boost", hint_boost),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"BKT parameter '{param_name}' must be in [0.0, 1.0], got {val}")

        self.p_init: float = p_init
        self.p_transit: float = p_transit
        self.p_slip: float = p_slip
        self.p_guess: float = p_guess
        self.hint_boost: float = hint_boost


class BKTEngine:
    """Bayesian Knowledge Tracing Engine maintaining belief states across multiple skills.

    Attributes:
        skills (List[str]): List of skill component names.
        params (Dict[str, BKTParams]): Mapping from skill name to BKT parameters.
        beliefs (np.ndarray): Current belief state vector P(L_t = 1) of shape (num_skills,).
    """

    DEFAULT_SKILLS: List[str] = [
        "Addition & Subtraction",
        "Multiplication & Division",
        "Algebra & Exponents",
    ]

    def __init__(
        self,
        skills: Optional[List[str]] = None,
        params: Optional[Dict[str, BKTParams]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes BKTEngine.

        Args:
            skills: Optional list of skill names. Defaults to DEFAULT_SKILLS.
            params: Optional dict of BKTParams per skill name.
            seed: Seed for random number generator.
        """
        self.skills: List[str] = list(skills) if skills else list(self.DEFAULT_SKILLS)
        self.num_skills: int = len(self.skills)
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Initialize parameters per skill
        self.params: Dict[str, BKTParams] = {}
        for skill in self.skills:
            if params and skill in params:
                self.params[skill] = params[skill]
            else:
                self.params[skill] = BKTParams()

        # Belief state vector b_t in [0, 1]^K
        self.beliefs: np.ndarray = np.zeros(self.num_skills, dtype=np.float32)
        self.reset()

    def reset(self, initial_beliefs: Optional[Union[List[float], np.ndarray]] = None) -> np.ndarray:
        """Resets belief states to initial priors.

        Args:
            initial_beliefs: Optional explicit initial belief array.

        Returns:
            np.ndarray: Initialized belief state vector of shape (num_skills,).
        """
        if initial_beliefs is not None:
            arr = np.array(initial_beliefs, dtype=np.float32)
            if arr.shape != (self.num_skills,):
                raise ValueError(f"initial_beliefs shape must be ({self.num_skills},), got {arr.shape}")
            self.beliefs = np.clip(arr, 0.0, 1.0)
        else:
            for idx, skill in enumerate(self.skills):
                p_init = self.params[skill].p_init
                # Add small noise around prior for diversity
                noise = float(self.rng.uniform(-0.05, 0.05))
                self.beliefs[idx] = float(np.clip(p_init + noise, 0.05, 0.95))

        return self.beliefs.copy()

    def update_belief(
        self,
        skill_idx: int,
        is_correct: bool,
        receives_scaffolding: bool = False,
    ) -> float:
        """Performs Bayesian posterior update and transition for a given skill.

        Args:
            skill_idx: Index of skill component in [0, num_skills - 1].
            is_correct: Observed answer correctness (True = Correct, False = Incorrect).
            receives_scaffolding: True if hint or worked example was studied.

        Returns:
            float: Updated belief P(L_{t+1} = 1) for the targeted skill.

        Raises:
            IndexError: If skill_idx is invalid.
        """
        if not (0 <= skill_idx < self.num_skills):
            raise IndexError(f"skill_idx must be in [0, {self.num_skills - 1}], got {skill_idx}")

        skill_name = self.skills[skill_idx]
        p = self.params[skill_name]
        b_t = float(self.beliefs[skill_idx])

        # 1. Posterior Likelihood Update (Bayes Rule)
        if is_correct:
            # P(O = 1 | L = 1) = 1 - P(S); P(O = 1 | L = 0) = P(G)
            numerator = b_t * (1.0 - p.p_slip)
            denominator = numerator + (1.0 - b_t) * p.p_guess
        else:
            # P(O = 0 | L = 1) = P(S); P(O = 0 | L = 0) = 1 - P(G)
            numerator = b_t * p.p_slip
            denominator = numerator + (1.0 - b_t) * (1.0 - p.p_guess)

        denominator = max(1e-8, denominator)
        p_posterior = numerator / denominator

        # 2. Transition / Learning Update
        p_transit_effective = p.p_transit + (p.hint_boost if receives_scaffolding else 0.0)
        p_transit_effective = min(1.0, p_transit_effective)

        b_next = p_posterior + (1.0 - p_posterior) * p_transit_effective
        b_next = float(np.clip(b_next, 0.0, 1.0))

        self.beliefs[skill_idx] = b_next
        return b_next

    def get_beliefs(self) -> np.ndarray:
        """Returns current belief state vector.

        Returns:
            np.ndarray: Vector of shape (num_skills,) with dtype float32.
        """
        return self.beliefs.copy()

    def get_mean_belief(self) -> float:
        """Returns mean belief across all skill components.

        Returns:
            float: Scalar mean knowledge belief in [0.0, 1.0].
        """
        return float(np.mean(self.beliefs))
