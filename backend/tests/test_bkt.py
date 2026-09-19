"""Unit tests for Bayesian Knowledge Tracing (BKT) and Memory Retention Decay."""

import numpy as np
import pytest
from app.core.exceptions import InvalidCognitiveStateException
from app.domain.cognitive.bkt import BKTEngine, BKTParams
from app.domain.cognitive.forgetting import RetentionDecayModel


def test_bkt_params_validation():
    """Ensures parameter bounds [0.0, 1.0] are strictly enforced."""
    with pytest.raises(InvalidCognitiveStateException):
        BKTParams(p_init=1.5)

    with pytest.raises(InvalidCognitiveStateException):
        BKTParams(p_slip=-0.1)


def test_bkt_posterior_update_correct():
    """Verifies Bayes' rule updates belief upward upon correct student responses."""
    bkt = BKTEngine(seed=42)
    initial_beliefs = bkt.get_beliefs()

    # Correct response on KC 0 without scaffolding
    new_b = bkt.update_belief(0, is_correct=True, receives_scaffolding=False)
    assert new_b > initial_beliefs[0], "Correct response must increase skill mastery belief"
    assert 0.0 < new_b < 1.0


def test_bkt_scaffolding_boost():
    """Verifies scaffolding hint boosts transition probability P(T)."""
    bkt_standard = BKTEngine(seed=123)
    b_no_scaff = bkt_standard.update_belief(1, is_correct=True, receives_scaffolding=False)

    bkt_scaffold = BKTEngine(seed=123)
    b_scaff = bkt_scaffold.update_belief(1, is_correct=True, receives_scaffolding=True)

    assert b_scaff > b_no_scaff, "Scaffolding hint must confer higher transition gain"


def test_retention_decay_and_urgency():
    """Verifies memory retention decay and review urgency calculation."""
    model = RetentionDecayModel(decay_rate=0.05, retention_floor=0.10)

    initial_b = 0.85
    decayed = model.calculate_decayed_belief(initial_b, steps_since_last_touch=10)
    assert decayed < initial_b, "Unvisited skill belief must decay over time"
    assert decayed >= 0.10, "Decayed belief must respect retention asymptote floor"

    beliefs = np.array([0.80, 0.20, 0.15, 0.10], dtype=np.float32)
    steps = [12, 1, 1, 1]
    urgency = model.compute_review_urgency(beliefs, steps)
    assert urgency > 0.40, "High mastery with many elapsed steps must trigger elevated review urgency"
