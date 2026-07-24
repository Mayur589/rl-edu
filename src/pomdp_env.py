"""POMDP Environment for Intelligent Tutoring System using BKT.

This module defines `POMDPTutorEnv`, a custom Gymnasium environment formulating
adaptive educational sequencing as a Partially Observable Markov Decision Process (POMDP).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.bkt_engine import BKTEngine


class POMDPTutorEnv(gym.Env):
    """POMDP Gymnasium Environment integrating Bayesian Knowledge Tracing (BKT).

    The environment simulates an unobservable true student mastery state across 3 math skills,
    exposing a belief state vector b_t computed via Bayesian Knowledge Tracing.

    Attributes:
        action_space (spaces.Discrete): 5 discrete pedagogical actions:
            0: Present Easy Problem (difficulty = 0.2)
            1: Present Medium Problem (difficulty = 0.5)
            2: Present Hard Problem (difficulty = 0.8)
            3: Present Worked Example (Scaffolding / Worked instruction)
            4: Provide Scaffolding Hint (Hint assistance)
        observation_space (spaces.Box): Continuous observation array of shape (8,):
            [0:3]: Skill belief state probabilities b_t^(0), b_t^(1), b_t^(2) in [0, 1]
            [3]: Mean belief state across all skills in [0, 1]
            [4]: Rolling accuracy over recent history in [0, 1]
            [5]: Response time / cognitive friction index in [0, 1]
            [6]: Normalized current step ratio (step / max_steps)
            [7]: Last action index normalized in [0, 1]
    """

    metadata: Dict[str, Any] = {"render_modes": ["human", "ansi"], "render_fps": 4}

    ACTION_MAP: Dict[int, str] = {
        0: "Easy Problem",
        1: "Medium Problem",
        2: "Hard Problem",
        3: "Worked Example",
        4: "Scaffolding Hint",
    }

    ACTION_DIFFICULTIES: Dict[int, float] = {
        0: 0.2,
        1: 0.5,
        2: 0.8,
        3: 0.3,
        4: 0.4,
    }

    def __init__(
        self,
        max_steps: int = 30,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initializes POMDPTutorEnv.

        Args:
            max_steps: Maximum time steps per episode. Must be > 0.
            render_mode: Optional render mode ('human' or 'ansi').

        Raises:
            ValueError: If max_steps <= 0.
        """
        super().__init__()

        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        self.max_steps: int = max_steps
        self.render_mode: Optional[str] = render_mode

        # BKT Engine for belief state tracking
        self.bkt_engine: BKTEngine = BKTEngine()
        self.num_skills: int = self.bkt_engine.num_skills  # 3 skills

        # 5 discrete pedagogical actions
        self.action_space: spaces.Discrete = spaces.Discrete(5)

        # 8-dimensional continuous observation vector
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_skills + 5,), dtype=np.float32
        )

        # Unobservable True Student Mastery Vector L_t in {0, 1}^K
        self.true_mastery: np.ndarray = np.zeros(self.num_skills, dtype=np.float32)

        # State tracking variables
        self.recent_history: List[float] = []
        self.last_action: int = 0
        self.last_correctness: float = 0.0
        self.last_friction: float = 0.0
        self.current_step: int = 0
        self.consecutive_hints: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment and BKT belief state.

        Args:
            seed: Seed for random number generator.
            options: Optional dict configurations.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)

        # Seed BKT engine
        self.bkt_engine = BKTEngine(seed=seed)
        initial_beliefs = self.bkt_engine.reset()

        # Initialize true hidden student state L_t (probabilistic sample from priors)
        self.true_mastery = (self.np_random.random(self.num_skills) < initial_beliefs).astype(np.float32)

        self.recent_history = []
        self.last_action = 0
        self.last_correctness = 0.0
        self.last_friction = 0.0
        self.current_step = 0
        self.consecutive_hints = 0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: Union[int, np.integer]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step in the POMDP environment.

        Args:
            action: Action index in {0, 1, 2, 3, 4}.

        Returns:
            Tuple containing:
                - observation: Updated 8D observation vector.
                - reward: Reward scalar calculated via Normalized Learning Gain (NLG).
                - terminated: True if all skill beliefs reach mastery (>= 0.95).
                - truncated: True if step reaches max_steps.
                - info: Detailed step metadata.

        Raises:
            ValueError: If action is not contained in action_space.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {list(range(5))}.")

        int_action = int(action)
        self.current_step += 1

        # Select target skill component (e.g. skill with lowest current belief)
        current_beliefs = self.bkt_engine.get_beliefs()
        target_skill_idx = int(np.argmin(current_beliefs))

        prev_mean_belief = self.bkt_engine.get_mean_belief()

        # Simulate response outcome based on action type
        is_scaffolding = int_action in (3, 4)  # Worked Example or Hint
        if int_action == 3:  # Worked Example
            # Worked example is studied; high engagement outcome
            is_correct = True
            friction = 0.2  # Low friction
            self.consecutive_hints += 1
        elif int_action == 4:  # Scaffolding Hint
            # Hint provided for problem; reduces slip probability
            target_diff = self.ACTION_DIFFICULTIES[4]
            true_l = self.true_mastery[target_skill_idx]
            p_correct = float(1.0 / (1.0 + np.exp(-8.0 * (true_l + 0.3 - target_diff))))
            is_correct = bool(self.np_random.random() < p_correct)
            friction = 0.5  # Moderate friction
            self.consecutive_hints += 1
        else:  # Practice Problems (Easy, Medium, Hard)
            self.consecutive_hints = 0
            difficulty = self.ACTION_DIFFICULTIES[int_action]
            true_l = self.true_mastery[target_skill_idx]
            # Item Response Theory accuracy model
            p_correct = float(1.0 / (1.0 + np.exp(-10.0 * (true_l - difficulty))))
            is_correct = bool(self.np_random.random() < p_correct)
            # Response friction higher on hard problems
            friction = min(1.0, 0.3 + 0.5 * difficulty + (0.3 if not is_correct else 0.0))

        correctness_val = 1.0 if is_correct else 0.0

        # Update BKT belief state
        new_skill_belief = self.bkt_engine.update_belief(
            skill_idx=target_skill_idx,
            is_correct=is_correct,
            receives_scaffolding=is_scaffolding,
        )

        # True latent state transition (probabilistic true learning)
        if is_correct or is_scaffolding:
            p_true_learn = 0.20 if is_scaffolding else 0.15
            if self.np_random.random() < p_true_learn:
                self.true_mastery[target_skill_idx] = 1.0

        new_mean_belief = self.bkt_engine.get_mean_belief()

        # Update recent rolling history
        self.recent_history.append(correctness_val)
        if len(self.recent_history) > 5:
            self.recent_history.pop(0)

        self.last_action = int_action
        self.last_correctness = correctness_val
        self.last_friction = friction

        # --- Calculate Reward via Normalized Learning Gain (NLG) ---
        # NLG = (b_{t+1} - b_t) / (1.0 - b_t)
        denom = max(1e-4, 1.0 - prev_mean_belief)
        nlg = (new_mean_belief - prev_mean_belief) / denom

        # Reward components
        reward_nlg = 10.0 * nlg
        penalty_step = -0.05  # Inefficiency penalty per step
        penalty_hint_spam = -0.3 if self.consecutive_hints > 2 else 0.0

        all_mastered = bool(np.all(current_beliefs >= 0.95))
        bonus_mastery = 3.0 if all_mastered else 0.0

        total_reward = float(reward_nlg + penalty_step + penalty_hint_spam + bonus_mastery)

        # Check termination & truncation
        terminated = all_mastered
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info.update(
            {
                "nlg": nlg,
                "is_correct": is_correct,
                "target_skill": target_skill_idx,
                "skill_name": self.bkt_engine.skills[target_skill_idx],
                "action_name": self.ACTION_MAP[int_action],
                "mean_belief": new_mean_belief,
            }
        )

        if self.render_mode == "human":
            self.render()

        return obs, total_reward, terminated, truncated, info

    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Renders current environment belief state.

        Returns:
            ANSI formatted string output.
        """
        beliefs = self.bkt_engine.get_beliefs()
        output = (
            f"Step: {self.current_step:2d}/{self.max_steps:2d} | "
            f"Action: {self.ACTION_MAP.get(self.last_action, 'N/A')} | "
            f"Mean Belief: {self.bkt_engine.get_mean_belief():.3f} | "
            f"Skill Beliefs: [{beliefs[0]:.2f}, {beliefs[1]:.2f}, {beliefs[2]:.2f}]"
        )
        if self.render_mode in ("human", "ansi"):
            print(output)
            return output
        return None

    def _get_obs(self) -> np.ndarray:
        """Constructs 8-dimensional observation vector.

        Returns:
            np.ndarray: Vector of shape (8,) with dtype float32.
        """
        beliefs = self.bkt_engine.get_beliefs()
        mean_b = self.bkt_engine.get_mean_belief()
        rolling_acc = float(np.mean(self.recent_history)) if self.recent_history else 0.0
        norm_step = min(self.current_step / float(self.max_steps), 1.0)
        norm_last_action = self.last_action / 4.0

        obs = np.array(
            [
                beliefs[0],
                beliefs[1],
                beliefs[2],
                mean_b,
                rolling_acc,
                self.last_friction,
                norm_step,
                norm_last_action,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Constructs step info dictionary.

        Returns:
            Dict[str, Any]: Environment state info.
        """
        return {
            "beliefs": self.bkt_engine.get_beliefs().tolist(),
            "mean_belief": self.bkt_engine.get_mean_belief(),
            "true_mastery": self.true_mastery.tolist(),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }
