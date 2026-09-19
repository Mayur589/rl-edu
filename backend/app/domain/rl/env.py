"""POMDP Gymnasium Pedagogical Environment for Adaptive Tutoring.

Formulates educational tutoring as a Partially Observable Markov Decision Process (POMDP):
- Hidden State: True cognitive mastery L_t* in {0, 1}^K
- Observable State: Bayesian Knowledge Tracing belief vector b_t, telemetry, and urgency
- Action Space: Content adaptation (Easy, Medium, Hard), Guidance adaptation (Hint, Worked Example), and Review adaptation (Spaced Review).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from app.core.config import settings
from app.domain.cognitive.bkt import BKTEngine
from app.domain.curriculum.repository import (
    KC_NAMES,
    QUESTION_BANK,
    check_prerequisites_met,
    get_questions_by_kc,
)
from app.domain.rl.reward import RewardCalculator, RewardComponents


class POMDPTutorEnv(gym.Env):
    """Gymnasium POMDP Environment for Intelligent Pedagogical Adaptation."""

    metadata: Dict[str, Any] = {"render_modes": ["human", "ansi"], "render_fps": 4}

    ACTION_MAP: Dict[int, str] = {
        0: "Easy Problem",
        1: "Medium Problem",
        2: "Hard Problem",
        3: "Scaffolding Hint",
        4: "Worked Example",
        5: "Spaced Review",
    }

    ACTION_DIFFICULTIES: Dict[int, float] = {
        0: 0.2,
        1: 0.5,
        2: 0.8,
        3: 0.3,
        4: 0.25,
        5: 0.4,
    }

    def __init__(
        self,
        max_steps: int = 30,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.bkt_engine = BKTEngine(seed=seed)
        self.num_skills = self.bkt_engine.num_skills  # 4 Knowledge Components
        self.reward_calculator = RewardCalculator()

        # 6 Discrete pedagogical actions
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.ACTION_MAP))

        # 10 Continuous observation dimensions:
        # [0:4] = 4 KC belief state probabilities
        # [4]   = Overall mean belief
        # [5]   = Rolling accuracy over recent attempts
        # [6]   = Cognitive friction / response latency index
        # [7]   = Normalized current step ratio (step / max_steps)
        # [8]   = Last action index normalized
        # [9]   = Spaced review urgency index
        self.obs_dim = self.num_skills + 6
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Unobservable Latent Student Mastery
        self.true_mastery: np.ndarray = np.zeros(self.num_skills, dtype=np.float32)

        # Interaction telemetry
        self.recent_history: List[float] = []
        self.last_action: int = 0
        self.last_correctness: float = 0.0
        self.last_friction: float = 0.0
        self.consecutive_hints: int = 0
        self.current_step: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment, BKT belief states, and student simulator."""
        super().reset(seed=seed)
        initial_beliefs = self.bkt_engine.reset()

        # Latent true mastery sampled probabilistically from priors
        self.true_mastery = (
            self.np_random.random(self.num_skills) < initial_beliefs
        ).astype(np.float32)

        self.recent_history = []
        self.last_action = 0
        self.last_correctness = 0.0
        self.last_friction = 0.0
        self.consecutive_hints = 0
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def select_target_kc(self, action: int) -> int:
        """Determines which KC to target based on prerequisite DAG and action type.

        If action == 5 (Review): target the skill with greatest review urgency.
        Otherwise: target the first eligible skill along the prerequisite DAG that has not reached mastery.
        """
        current_beliefs = self.bkt_engine.get_beliefs()

        if action == 5:
            # Spaced review targets the KC with highest elapsed steps and non-trivial mastery
            steps = self.bkt_engine.steps_since_touch
            review_candidate = int(np.argmax([s * (b > 0.3) for s, b in zip(steps, current_beliefs)]))
            return review_candidate

        # Content/Guidance adaptation: find earliest unmastered KC with prerequisites satisfied
        for kc_idx in range(self.num_skills):
            if current_beliefs[kc_idx] < 0.88:
                if check_prerequisites_met(kc_idx, current_beliefs.tolist(), threshold=0.65):
                    return kc_idx

        # If all prerequisites met, pick the lowest belief
        return int(np.argmin(current_beliefs))

    def step(
        self, action: Union[int, np.integer]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single pedagogical transition step in the POMDP environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {list(range(len(self.ACTION_MAP)))}")

        int_action = int(action)
        self.current_step += 1

        prev_mean_belief = self.bkt_engine.get_mean_belief()
        review_urgency = self.bkt_engine.get_review_urgency()
        target_kc = self.select_target_kc(int_action)

        # 1. Determine student response dynamics via Item Response Theory (IRT)
        is_scaffolding = int_action in (3, 4)
        if int_action == 4:  # Worked Example
            is_correct = True
            friction = 0.15
            self.consecutive_hints += 1
        elif int_action == 3:  # Scaffolding Hint
            target_diff = self.ACTION_DIFFICULTIES[3]
            true_l = self.true_mastery[target_kc]
            # Hint increases log-odds of success
            p_correct = float(1.0 / (1.0 + np.exp(-8.0 * (true_l + 0.35 - target_diff))))
            is_correct = bool(self.np_random.random() < p_correct)
            friction = 0.45
            self.consecutive_hints += 1
        elif int_action == 5:  # Spaced Review
            self.consecutive_hints = 0
            difficulty = self.ACTION_DIFFICULTIES[5]
            true_l = self.true_mastery[target_kc]
            p_correct = float(1.0 / (1.0 + np.exp(-8.0 * (true_l - difficulty))))
            is_correct = bool(self.np_random.random() < p_correct)
            friction = 0.35
        else:  # Practice Problems (0: Easy, 1: Medium, 2: Hard)
            self.consecutive_hints = 0
            difficulty = self.ACTION_DIFFICULTIES[int_action]
            true_l = self.true_mastery[target_kc]
            p_correct = float(1.0 / (1.0 + np.exp(-9.0 * (true_l - difficulty))))
            is_correct = bool(self.np_random.random() < p_correct)
            friction = float(np.clip(0.25 + 0.55 * difficulty + (0.25 if not is_correct else 0.0), 0.0, 1.0))

        correctness_val = 1.0 if is_correct else 0.0

        # 2. Update BKT belief distribution
        self.bkt_engine.update_belief(
            skill_idx=target_kc,
            is_correct=is_correct,
            receives_scaffolding=is_scaffolding,
        )

        # 3. Simulate latent cognitive transition (true student learning)
        if is_correct or is_scaffolding:
            p_true_learning = 0.25 if is_scaffolding else 0.18
            if self.np_random.random() < p_true_learning:
                self.true_mastery[target_kc] = 1.0

        new_mean_belief = self.bkt_engine.get_mean_belief()

        # Update telemetry history
        self.recent_history.append(correctness_val)
        if len(self.recent_history) > 5:
            self.recent_history.pop(0)

        self.last_action = int_action
        self.last_correctness = correctness_val
        self.last_friction = friction

        # 4. Termination conditions
        all_mastered = bool(np.all(self.bkt_engine.get_beliefs() >= 0.88))
        terminated = all_mastered
        truncated = bool(self.current_step >= self.max_steps)

        # 5. Multi-Objective Reward Decomposition
        reward_decomp: RewardComponents = self.reward_calculator.calculate(
            prev_mean_belief=prev_mean_belief,
            new_mean_belief=new_mean_belief,
            consecutive_hints=self.consecutive_hints,
            friction=friction,
            action=int_action,
            review_urgency=review_urgency,
            is_all_mastered=all_mastered,
        )

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "target_kc": target_kc,
            "target_kc_name": KC_NAMES[target_kc],
            "is_correct": is_correct,
            "friction": friction,
            "reward_breakdown": reward_decomp.to_dict(),
        })

        return obs, reward_decomp.total_reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Constructs the 10-dimensional continuous observation vector."""
        beliefs = self.bkt_engine.get_beliefs()
        mean_belief = self.bkt_engine.get_mean_belief()
        rolling_acc = float(np.mean(self.recent_history)) if self.recent_history else 0.5
        friction = self.last_friction
        step_ratio = float(self.current_step / max(1, self.max_steps))
        last_action_norm = float(self.last_action / max(1, len(self.ACTION_MAP) - 1))
        urgency = self.bkt_engine.get_review_urgency()

        obs = np.array(
            [
                beliefs[0],
                beliefs[1],
                beliefs[2],
                beliefs[3],
                mean_belief,
                rolling_acc,
                friction,
                step_ratio,
                last_action_norm,
                urgency,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        """Provides metadata on current environment state for logging and UI."""
        return {
            "current_step": self.current_step,
            "mean_belief": round(self.bkt_engine.get_mean_belief(), 4),
            "beliefs": [round(b, 4) for b in self.bkt_engine.get_beliefs().tolist()],
            "true_mastery": self.true_mastery.tolist(),
            "last_action": self.last_action,
            "last_action_name": self.ACTION_MAP.get(self.last_action, "Unknown"),
            "consecutive_hints": self.consecutive_hints,
            "review_urgency": round(self.bkt_engine.get_review_urgency(), 4),
        }
