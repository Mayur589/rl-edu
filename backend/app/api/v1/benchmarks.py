"""Benchmarking API Endpoints."""

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.db.models import User
from app.domain.rl.agent import D3QNAgent
from app.domain.rl.evaluator import BenchmarkEvaluator
from app.schemas.analytics import BenchmarkPolicyComparison, BenchmarkResultResponse, BenchmarkRunRequest

router = APIRouter(prefix="/benchmarks", tags=["Benchmarking"])


@router.post("/run", response_model=BenchmarkResultResponse)
def run_benchmark(
    request: BenchmarkRunRequest,
    current_user: User = Depends(get_current_user),
):
    """Executes comparative Monte Carlo benchmark comparing D3QN vs standard pedagogical baselines."""
    evaluator = BenchmarkEvaluator(
        max_steps=request.max_steps, episodes=request.episodes, seed=42
    )
    # Instantiate fresh D3QN agent with quick pretraining
    agent = D3QNAgent(seed=42)
    res = evaluator.run_full_benchmark(trained_agent=agent)

    comparison_dtos = [
        BenchmarkPolicyComparison(
            baseline=item["baseline"],
            nlg=item["nlg"],
            reward=item["reward"],
            mastery_rate=item["mastery_rate"],
            cohens_d=item["cohens_d"],
            significant=item["significant"],
        )
        for item in res["comparison_table"]
    ]

    return BenchmarkResultResponse(
        rl_agent_summary=res["rl_agent_summary"],
        all_policies=res["all_policies"],
        comparison_table=comparison_dtos,
    )
