"""Simulated Student Environment for Reinforcement Learning.

This module defines `SimulatedStudentEnv`, a custom Gymnasium environment
formulating adaptive learning as a Markov Decision Process (MDP).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimulatedStudentEnv(gym.Env):
    """Custom Gymnasium Environment simulating a student's learning trajectory.

    The environment models student dynamics using an Item Response Theory (IRT)
    probabilistic framework. An RL agent selects problem difficulty levels (actions),
    and receives rewards based on learning gains, answer accuracy, and alignment
    with the student's Zone of Proximal Development (ZPD).

    Attributes:
        action_space (spaces.Discrete): Discrete action space with 3 difficulty levels:
            0: Easy (difficulty = 0.2)
            1: Medium (difficulty = 0.5)
            2: Hard (difficulty = 0.8)
        observation_space (spaces.Box): Continuous observation vector of shape (7,):
            [0]: Latent knowledge level theta in [0.0, 1.0]
            [1]: Last question difficulty in [0.0, 1.0]
            [2]: Last answer accuracy (0.0 or 1.0)
            [3]: Rolling accuracy over recent window in [0.0, 1.0]
            [4]: Normalized consecutive correct answers
            [5]: Normalized consecutive incorrect answers
            [6]: Normalized current step count in episode
        metadata (Dict[str, Any]): Gymnasium environment metadata rendering options.
    """

    metadata: Dict[str, Any] = {"render_modes": ["human", "ansi"], "render_fps": 4}

    DIFFICULTY_MAP: Dict[int, float] = {
        0: 0.2,  # Easy
        1: 0.5,  # Medium
        2: 0.8,  # Hard
    }

    DIFFICULTY_NAMES: Dict[int, str] = {
        0: "Easy",
        1: "Medium",
        2: "Hard",
    }

    def __init__(
        self,
        max_steps: int = 30,
        initial_knowledge: Optional[float] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initializes the SimulatedStudentEnv.

        Args:
            max_steps: Maximum number of questions per episode. Must be > 0.
            initial_knowledge: Fixed initial knowledge state theta in [0.0, 1.0].
                If None, initial knowledge is sampled uniformly from [0.1, 0.3].
            render_mode: Render mode ('human' or 'ansi').

        Raises:
            ValueError: If max_steps <= 0 or initial_knowledge is outside [0.0, 1.0].
        """
        super().__init__()

        if max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer, got {max_steps}")
        if initial_knowledge is not None and not (0.0 <= initial_knowledge <= 1.0):
            raise ValueError(
                f"initial_knowledge must be between 0.0 and 1.0, got {initial_knowledge}"
            )

        self.max_steps: int = max_steps
        self.fixed_initial_knowledge: Optional[float] = initial_knowledge
        self.render_mode: Optional[str] = render_mode

        # Define 3 discrete difficulty choices: Easy (0), Medium (1), Hard (2)
        self.action_space: spaces.Discrete = spaces.Discrete(3)

        # Observation vector: 7 continuous features normalized between 0.0 and 1.0
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # State variables
        self.student_knowledge: float = 0.2
        self.last_difficulty: float = 0.0
        self.last_correctness: float = 0.0
        self.recent_history: list[float] = []
        self.consecutive_correct: int = 0
        self.consecutive_incorrect: int = 0
        self.current_step: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state.

        Args:
            seed: Seed for setting random number generator state.
            options: Optional configuration dictionary. May include 'initial_knowledge'.

        Returns:
            Tuple containing:
                - observation: Initial observation array of shape (7,).
                - info: Diagnostic info dictionary.
        """
        super().reset(seed=seed)

        # Allow initial knowledge override via options
        opt_init = options.get("initial_knowledge") if options else None
        if opt_init is not None:
            if not (0.0 <= opt_init <= 1.0):
                raise ValueError(f"Option initial_knowledge must be in [0.0, 1.0], got {opt_init}")
            self.student_knowledge = float(opt_init)
        elif self.fixed_initial_knowledge is not None:
            self.student_knowledge = float(self.fixed_initial_knowledge)
        else:
            # Sample initial knowledge from uniform range [0.1, 0.3]
            self.student_knowledge = float(self.np_random.uniform(0.1, 0.3))

        self.last_difficulty = 0.0
        self.last_correctness = 0.0
        self.recent_history = []
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: Union[int, np.integer]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step in the environment.

        Args:
            action: Selected question difficulty level (0: Easy, 1: Medium, 2: Hard).

        Returns:
            Tuple containing:
                - observation: Observation vector after step execution.
                - reward: Scalar reward calculated from learning gain and ZPD alignment.
                - terminated: True if student achieves mastery (knowledge >= 0.95).
                - truncated: True if episode reaches max_steps.
                - info: Diagnostic dictionary with step metadata.

        Raises:
            ValueError: If action is not a valid discrete action in space.
        """
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. Must be an integer in {list(self.DIFFICULTY_MAP.keys())}."
            )

        int_action = int(action)
        difficulty = self.DIFFICULTY_MAP[int_action]
        self.current_step += 1

        # 1. Item Response Theory (IRT) probability of correct answer
        # Alpha parameter governs discrimination curve steepness
        alpha = 10.0
        logit = alpha * (self.student_knowledge - difficulty)
        p_correct = float(1.0 / (1.0 + np.exp(-logit)))

        # Sample actual student response
        is_correct = bool(self.np_random.random() < p_correct)
        correctness_val = 1.0 if is_correct else 0.0

        # 2. Update Student Knowledge State
        prev_knowledge = self.student_knowledge
        if is_correct:
            # Learning gain proportional to difficulty challenge
            learning_gain = 0.15 * (1.0 - self.student_knowledge) * difficulty
            self.student_knowledge = min(1.0, self.student_knowledge + learning_gain)
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
        else:
            # Small consolidation gain even on incorrect attempts
            attempt_gain = 0.02 * (1.0 - self.student_knowledge)
            self.student_knowledge = min(1.0, self.student_knowledge + attempt_gain)
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0

        delta_knowledge = self.student_knowledge - prev_knowledge

        # Update rolling history (keep last 5 outcomes)
        self.recent_history.append(correctness_val)
        if len(self.recent_history) > 5:
            self.recent_history.pop(0)

        self.last_difficulty = difficulty
        self.last_correctness = correctness_val

        # 3. Calculate Reward
        # Reward component A: Direct learning gain
        reward_gain = 10.0 * delta_knowledge

        # Reward component B: Zone of Proximal Development (ZPD) alignment penalty
        # Ideal difficulty matches current knowledge level
        zpd_penalty = -0.5 * float((difficulty - prev_knowledge) ** 2)

        # Reward component C: Success encouragement bonus
        accuracy_bonus = 0.1 * difficulty if is_correct else 0.0

        # Reward component D: Mastery bonus
        mastery_achieved = self.student_knowledge >= 0.95
        mastery_bonus = 2.0 if mastery_achieved else 0.0

        total_reward = float(reward_gain + zpd_penalty + accuracy_bonus + mastery_bonus)

        # 4. Check Termination & Truncation
        terminated = mastery_achieved
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["p_correct"] = p_correct
        info["is_correct"] = is_correct
        info["delta_knowledge"] = delta_knowledge
        info["difficulty"] = difficulty
        info["difficulty_name"] = self.DIFFICULTY_NAMES[int_action]

        if self.render_mode == "human":
            self.render()

        return obs, total_reward, terminated, truncated, info

    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Renders the current environment state.

        Returns:
            ANSI formatted string summary if render_mode is 'ansi' or 'human'.
        """
        output = (
            f"Step: {self.current_step:2d}/{self.max_steps:2d} | "
            f"Knowledge: {self.student_knowledge:.3f} | "
            f"Last Diff: {self.last_difficulty:.1f} | "
            f"Last Correct: {int(self.last_correctness)} | "
            f"Streak: +{self.consecutive_correct}/-{self.consecutive_incorrect}"
        )
        if self.render_mode in ("human", "ansi"):
            print(output)
            return output
        return None

    def _get_obs(self) -> np.ndarray:
        """Constructs and returns the 7-dimensional state observation vector.

        Returns:
            np.ndarray: Observation vector of shape (7,) with dtype float32.
        """
        rolling_acc = (
            float(np.mean(self.recent_history)) if self.recent_history else 0.0
        )
        norm_correct_streak = min(self.consecutive_correct / 5.0, 1.0)
        norm_incorrect_streak = min(self.consecutive_incorrect / 5.0, 1.0)
        norm_step = min(self.current_step / float(self.max_steps), 1.0)

        obs = np.array(
            [
                self.student_knowledge,
                self.last_difficulty,
                self.last_correctness,
                rolling_acc,
                norm_correct_streak,
                norm_incorrect_streak,
                norm_step,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Constructs diagnostic information dictionary.

        Returns:
            Dict[str, Any]: Environment state details.
        """
        return {
            "student_knowledge": self.student_knowledge,
            "consecutive_correct": self.consecutive_correct,
            "consecutive_incorrect": self.consecutive_incorrect,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }
