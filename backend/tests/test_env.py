"""Unit tests for POMDP Tutor Gymnasium Environment."""

import numpy as np
import pytest
from app.domain.rl.env import POMDPTutorEnv


def test_pomdp_env_init():
    """Verifies observation and action space specifications."""
    env = POMDPTutorEnv(max_steps=20, seed=42)
    assert env.action_space.n == 6
    assert env.observation_space.shape == (10,)
    assert env.observation_space.low[0] == 0.0
    assert env.observation_space.high[0] == 1.0


def test_pomdp_env_step_cycle():
    """Verifies environment transitions, observation clipping, and reward breakdown."""
    env = POMDPTutorEnv(max_steps=10, seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (10,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert "mean_belief" in info
    assert "review_urgency" in info

    # Step through all 6 actions
    for action in range(6):
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs.shape == (10,)
        assert np.all(next_obs >= 0.0) and np.all(next_obs <= 1.0)
        assert isinstance(reward, float)
        assert "reward_breakdown" in step_info
        assert "nlg_reward" in step_info["reward_breakdown"]
        assert "step_penalty" in step_info["reward_breakdown"]


def test_invalid_action():
    """Ensures invalid actions raise ValueError."""
    env = POMDPTutorEnv(max_steps=10)
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)
