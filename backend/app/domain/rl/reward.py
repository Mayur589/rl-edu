"""Multi-Objective Reward Decomposition for POMDP Pedagogical Agent.

Implements transparent, explainable reward components:
1. Normalized Learning Gain (NLG) reward
2. Efficiency step penalty
3. Scaffolding / Hint spamming penalty
4. Cognitive friction penalty
5. Spaced review alignment bonus
6. Terminal mastery bonus
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardComponents:
    """Decomposed scalar reward elements for explainability."""

    nlg_reward: float
    step_penalty: float
    hint_penalty: float
    friction_penalty: float
    review_bonus: float
    mastery_bonus: float
    total_reward: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "nlg_reward": round(self.nlg_reward, 4),
            "step_penalty": round(self.step_penalty, 4),
            "hint_penalty": round(self.hint_penalty, 4),
            "friction_penalty": round(self.friction_penalty, 4),
            "review_bonus": round(self.review_bonus, 4),
            "mastery_bonus": round(self.mastery_bonus, 4),
            "total_reward": round(self.total_reward, 4),
        }


class RewardCalculator:
    """Calculates multi-objective rewards with pedagogical objective weights."""

    def __init__(
        self,
        nlg_weight: float = 5.0,
        step_cost: float = 0.40,
        hint_penalty_cost: float = 0.80,
        friction_weight: float = 0.30,
        review_bonus_weight: float = 1.50,
        mastery_bonus_value: float = 50.0,
    ) -> None:
        self.nlg_weight = nlg_weight
        self.step_cost = step_cost
        self.hint_penalty_cost = hint_penalty_cost
        self.friction_weight = friction_weight
        self.review_bonus_weight = review_bonus_weight
        self.mastery_bonus_value = mastery_bonus_value

    def calculate(
        self,
        prev_mean_belief: float,
        new_mean_belief: float,
        consecutive_hints: int,
        friction: float,
        action: int,
        review_urgency: float,
        is_all_mastered: bool,
    ) -> RewardComponents:
        """Computes decomposed reward components for state transition."""
        # 1. Normalized Learning Gain (NLG)
        denom = max(1e-4, 1.0 - prev_mean_belief)
        nlg = (new_mean_belief - prev_mean_belief) / denom
        nlg_reward = self.nlg_weight * nlg

        # 2. Step cost (penalize excessive time-to-mastery)
        step_penalty = -self.step_cost

        # 3. Hint dependency penalty (if student/agent spams hints > 2 consecutive times)
        hint_penalty = -self.hint_penalty_cost if consecutive_hints > 2 else 0.0

        # 4. Cognitive friction penalty
        friction_penalty = -self.friction_weight * friction

        # 5. Spaced review incentive (reward reviewing when urgency is high, penalize unnecessary review)
        if action == 5:  # Review action
            if review_urgency > 0.4:
                review_bonus = self.review_bonus_weight * review_urgency
            else:
                review_bonus = -0.50  # Premature review when fresh
        else:
            review_bonus = 0.0

        # 6. Global curriculum mastery bonus
        mastery_bonus = self.mastery_bonus_value if is_all_mastered else 0.0

        total = float(
            nlg_reward
            + step_penalty
            + hint_penalty
            + friction_penalty
            + review_bonus
            + mastery_bonus
        )

        return RewardComponents(
            nlg_reward=float(nlg_reward),
            step_penalty=float(step_penalty),
            hint_penalty=float(hint_penalty),
            friction_penalty=float(friction_penalty),
            review_bonus=float(review_bonus),
            mastery_bonus=float(mastery_bonus),
            total_reward=total,
        )
