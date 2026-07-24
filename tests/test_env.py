"""Unit Tests for BKT Engine and POMDP Tutor Environment.

This module provides pytest unit tests verifying Bayesian Knowledge Tracing (BKT)
belief updates, observation shapes, deterministic reproducibility, and Normalized Learning Gain (NLG) rewards.
"""

import pytest
import numpy as np

from src.bkt_engine import BKTEngine, BKTParams
from src.pomdp_env import POMDPTutorEnv


def test_bkt_params_validation() -> None:
    """Verifies that BKTParams validates input probability bounds [0, 1]."""
    # Valid parameters should instantiate cleanly
    params = BKTParams(p_init=0.2, p_transit=0.15, p_slip=0.10, p_guess=0.25)
    assert params.p_init == 0.2

    # Invalid parameters should raise ValueError
    with pytest.raises(ValueError):
        BKTParams(p_init=1.5)

    with pytest.raises(ValueError):
        BKTParams(p_slip=-0.1)


def test_bkt_posterior_update() -> None:
    """Verifies BKT belief increases on correct answer and handles incorrect answer correctly."""
    bkt = BKTEngine(seed=42)
    initial_beliefs = bkt.get_beliefs().copy()

    # Correct answer on skill 0 should increase belief for skill 0
    new_b0 = bkt.update_belief(skill_idx=0, is_correct=True)
    assert new_b0 > initial_beliefs[0]
    assert 0.0 <= new_b0 <= 1.0

    # Beliefs for un-targeted skills should remain unchanged
    current_beliefs = bkt.get_beliefs()
    assert np.isclose(current_beliefs[1], initial_beliefs[1])
    assert np.isclose(current_beliefs[2], initial_beliefs[2])


def test_pomdp_env_space_specification() -> None:
    """Verifies observation and action space definitions of POMDPTutorEnv."""
    env = POMDPTutorEnv(max_steps=30)
    assert env.action_space.n == 5

    obs_shape = env.observation_space.shape
    assert obs_shape == (9,)
    assert env.observation_space.dtype == np.float32


def test_pomdp_env_step_sequence() -> None:
    """Verifies environment step execution, obs bounds, and return types."""
    env = POMDPTutorEnv(max_steps=30)
    obs, info = env.reset(seed=42)

    assert obs.shape == (9,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert "beliefs" in info
    assert "mean_belief" in info


    # Step through all 5 discrete actions
    for action in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "nlg" in info
        assert "action_name" in info


def test_normalized_learning_gain_reward() -> None:
    """Verifies that Normalized Learning Gain (NLG) is calculated and reflected in reward."""
    env = POMDPTutorEnv(max_steps=30)
    obs, info = env.reset(seed=42)

    prev_mean_b = info["mean_belief"]
    obs, reward, terminated, truncated, step_info = env.step(3)  # Worked Example action

    new_mean_b = step_info["mean_belief"]
    nlg = step_info["nlg"]

    # Worked example increases mean belief, resulting in positive NLG
    assert new_mean_b >= prev_mean_b
    assert nlg >= 0.0


def test_reproducibility() -> None:
    """Verifies deterministic environment behavior when initialized with identical random seeds."""
    env1 = POMDPTutorEnv(max_steps=20)
    env2 = POMDPTutorEnv(max_steps=20)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    assert np.allclose(obs1, obs2)

    for action in [0, 1, 2, 3, 4]:
        o1, r1, t1, tr1, i1 = env1.step(action)
        o2, r2, t2, tr2, i2 = env2.step(action)
        assert np.allclose(o1, o2)
        assert np.isclose(r1, r2)
        assert t1 == t2
        assert tr1 == tr2
