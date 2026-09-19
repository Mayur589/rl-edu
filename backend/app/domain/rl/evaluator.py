"""Evaluation and Benchmarking Suite for Pedagogical Policies.

Runs multi-episode Monte Carlo evaluations across RL Agent and Baseline policies,
computing Normalized Learning Gain (NLG), Time-to-Mastery, Reward distributions,
and effect size statistics (Cohen's d).
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats

from app.domain.rl.agent import D3QNAgent
from app.domain.rl.baselines import (
    BasePedagogicalPolicy,
    LeitnerSpacedRepetitionPolicy,
    LinearCurriculumPolicy,
    RandomPolicy,
    RuleBasedHeuristicPolicy,
)
from app.domain.rl.env import POMDPTutorEnv


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculates Cohen's d effect size between two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = float(np.mean(group1)), float(np.mean(group2))
    var1, var2 = float(np.var(group1, ddof=1)), float(np.var(group2, ddof=1))

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-8:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


class BenchmarkEvaluator:
    """Runs rigorous multi-episode policy comparisons."""

    def __init__(self, max_steps: int = 30, episodes: int = 25, seed: int = 42) -> None:
        self.max_steps = max_steps
        self.episodes = episodes
        self.seed = seed

    def evaluate_policy(
        self,
        policy: Any,
        policy_name: str,
        is_rl_agent: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates a policy over multiple simulated student episodes."""
        env = POMDPTutorEnv(max_steps=self.max_steps, seed=self.seed)

        rewards: List[float] = []
        nlgs: List[float] = []
        steps_list: List[int] = []
        frictions: List[float] = []
        mastered_rates: List[float] = []

        for ep in range(self.episodes):
            ep_seed = self.seed + ep * 100
            obs, info = env.reset(seed=ep_seed)
            b_initial = info["mean_belief"]
            ep_reward = 0.0
            ep_frictions = []
            done = False

            while not done:
                if is_rl_agent:
                    action = policy.select_action(obs, evaluate=True)
                else:
                    action = policy.select_action(obs)

                obs, r, term, trunc, step_info = env.step(action)
                ep_reward += r
                ep_frictions.append(step_info["friction"])
                done = term or trunc

            b_final = step_info["mean_belief"]
            denom = max(1e-4, 1.0 - b_initial)
            nlg = (b_final - b_initial) / denom

            rewards.append(ep_reward)
            nlgs.append(float(nlg))
            steps_list.append(step_info["current_step"])
            frictions.append(float(np.mean(ep_frictions)))
            mastered_rates.append(1.0 if term else 0.0)

        return {
            "policy_name": policy_name,
            "episodes": self.episodes,
            "mean_reward": round(float(np.mean(rewards)), 3),
            "std_reward": round(float(np.std(rewards)), 3),
            "mean_nlg": round(float(np.mean(nlgs)), 4),
            "std_nlg": round(float(np.std(nlgs)), 4),
            "mean_steps": round(float(np.mean(steps_list)), 2),
            "mean_friction": round(float(np.mean(frictions)), 3),
            "mastery_rate": round(float(np.mean(mastered_rates)) * 100, 1),
            "raw_rewards": rewards,
            "raw_nlgs": nlgs,
        }

    def run_full_benchmark(
        self, trained_agent: Optional[D3QNAgent] = None
    ) -> Dict[str, Any]:
        """Runs comparative evaluation against all standard baselines."""
        policies = [
            ("Random Baseline", RandomPolicy(seed=self.seed), False),
            ("Linear Curriculum", LinearCurriculumPolicy(), False),
            ("Rule-Based Heuristic", RuleBasedHeuristicPolicy(), False),
            ("Leitner Spaced Repetition", LeitnerSpacedRepetitionPolicy(), False),
        ]

        # Initialize or use provided RL agent
        if trained_agent is None:
            agent = D3QNAgent(seed=self.seed)
            # Quick pre-training
            env = POMDPTutorEnv(max_steps=self.max_steps, seed=self.seed)
            for _ in range(15):
                s, _ = env.reset()
                d = False
                while not d:
                    a = agent.select_action(s)
                    sn, r, t, tr, _ = env.step(a)
                    d = t or tr
                    agent.buffer.push(s, a, r, sn, d)
                    agent.train_step()
                    s = sn
        else:
            agent = trained_agent

        results: List[Dict[str, Any]] = []

        # 1. Evaluate RL Agent
        rl_res = self.evaluate_policy(agent, "Explainable D3QN (Ours)", is_rl_agent=True)
        results.append(rl_res)

        # 2. Evaluate Baselines and calculate statistical effect sizes
        comparison_table = []
        for name, pol, is_rl in policies:
            res = self.evaluate_policy(pol, name, is_rl_agent=is_rl)
            # Compute statistical tests vs RL Agent
            t_stat, p_val = stats.ttest_ind(rl_res["raw_nlgs"], res["raw_nlgs"], equal_var=False)
            d_val = compute_cohens_d(rl_res["raw_nlgs"], res["raw_nlgs"])

            res["p_value_vs_rl"] = round(float(p_val), 4)
            res["cohens_d_vs_rl"] = round(float(d_val), 3)
            results.append(res)

            comparison_table.append({
                "baseline": name,
                "nlg": res["mean_nlg"],
                "reward": res["mean_reward"],
                "mastery_rate": f"{res['mastery_rate']}%",
                "cohens_d": round(float(d_val), 3),
                "significant": bool(p_val < 0.05),
            })

        return {
            "rl_agent_summary": {
                "name": rl_res["policy_name"],
                "mean_nlg": rl_res["mean_nlg"],
                "mean_reward": rl_res["mean_reward"],
                "mastery_rate": f"{rl_res['mastery_rate']}%",
            },
            "all_policies": [
                {k: v for k, v in r.items() if not k.startswith("raw_")}
                for r in results
            ],
            "comparison_table": comparison_table,
        }
