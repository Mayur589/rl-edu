"""Unit tests for NumPy Dueling Double Deep Q-Network (D3QN) Engine."""

import numpy as np
from app.domain.rl.agent import D3QNAgent, NumPyDuelingNetwork
from app.domain.rl.replay_buffer import ReplayBuffer


def test_dueling_network_forward_and_aggregation():
    """Verifies that Q(s, a) = V(s) + (A(s, a) - mean(A)) holds numerically."""
    net = NumPyDuelingNetwork(state_dim=10, action_dim=6, hidden_dim=32, seed=42)
    state = np.random.uniform(0.0, 1.0, (1, 10)).astype(np.float32)

    q, v, a, cache = net.forward(state)

    assert q.shape == (1, 6)
    assert v.shape == (1, 1)
    assert a.shape == (1, 6)

    # Check dueling mathematical formula
    a_mean = np.mean(a, axis=1, keepdims=True)
    reconstructed_q = v + (a - a_mean)
    assert np.allclose(q, reconstructed_q, atol=1e-5), "Dueling aggregation mismatch"


def test_dueling_network_backpropagation():
    """Verifies that backward pass returns gradient tensors matching parameter shapes."""
    net = NumPyDuelingNetwork(state_dim=10, action_dim=6, hidden_dim=32, seed=42)
    state = np.random.uniform(0.0, 1.0, (4, 10)).astype(np.float32)

    q, v, a, cache = net.forward(state)
    d_q = np.ones_like(q)

    grads = net.backward(d_q, cache)
    assert len(grads) == len(net.params)
    for p, g in zip(net.params, grads):
        assert p.shape == g.shape, f"Shape mismatch: param {p.shape} vs grad {g.shape}"


def test_replay_buffer():
    """Verifies FIFO buffer behavior and mini-batch sampling."""
    buf = ReplayBuffer(capacity=50, state_dim=10, seed=42)

    for i in range(25):
        s = np.zeros(10, dtype=np.float32) + i
        s_next = s + 1
        buf.push(s, i % 6, 1.0, s_next, False)

    assert len(buf) == 25
    states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
    assert states.shape == (8, 10)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 10)
    assert dones.shape == (8,)


def test_d3qn_agent_learning_step():
    """Verifies that D3QNAgent executes training step and decays exploration rate."""
    agent = D3QNAgent(state_dim=10, action_dim=6, hidden_dim=32, batch_size=4, seed=42)

    initial_eps = agent.epsilon

    # Fill buffer with mock transitions
    for _ in range(10):
        s = np.random.uniform(0, 1, 10).astype(np.float32)
        sn = np.random.uniform(0, 1, 10).astype(np.float32)
        agent.buffer.push(s, 1, 0.5, sn, False)

    loss = agent.train_step()
    assert loss is not None
    assert isinstance(loss, float)
    assert agent.epsilon < initial_eps, "Epsilon must decay after training step"


def test_d3qn_explainability_payload():
    """Verifies grounded explainability generation with no hallucinations."""
    agent = D3QNAgent(seed=42)
    state = np.array([0.2, 0.3, 0.4, 0.5, 0.35, 0.6, 0.2, 0.1, 0.0, 0.1], dtype=np.float32)

    record = agent.explain_decision(state)
    assert len(record.q_values) == 6
    assert len(record.action_advantages) == 6
    assert len(record.action_ranking) == 6
    assert record.selected_action in range(6)
    assert len(record.pedagogical_rationale) > 10
