"""Unit tests for Benchmark Evaluator and Statistical Tests."""

from app.domain.rl.baselines import LinearCurriculumPolicy, RandomPolicy, RuleBasedHeuristicPolicy
from app.domain.rl.evaluator import BenchmarkEvaluator, compute_cohens_d


def test_cohens_d_computation():
    g1 = [10.0, 11.0, 12.0, 10.5, 11.5]
    g2 = [5.0, 6.0, 5.5, 4.5, 6.5]
    d = compute_cohens_d(g1, g2)
    assert d > 2.0, f"Expected large effect size d > 2, got {d}"


def test_benchmark_evaluator_fast():
    evaluator = BenchmarkEvaluator(max_steps=5, episodes=2, seed=42)
    results = evaluator.evaluate_policy(RuleBasedHeuristicPolicy(), "Heuristic", is_rl_agent=False)
    assert results["episodes"] == 2
    assert "mean_nlg" in results
    assert "mean_reward" in results
