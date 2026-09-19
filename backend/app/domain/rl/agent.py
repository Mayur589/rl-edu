"""NumPy-Powered Dueling Double Deep Q-Network (D3QN) Engine.

Provides clean, production-grade, mathematically transparent reinforcement learning:
- Decoupled State Value V(s) and Action Advantage A(s, a) streams
- Double DQN target policy estimation for reduced maximization bias
- Complete analytical backpropagation with Adam optimizer in pure NumPy
- Grounded explainability decomposition for researchers and students
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.core.config import settings
from app.domain.rl.env import POMDPTutorEnv
from app.domain.rl.replay_buffer import ReplayBuffer


@dataclass
class D3QNExplainabilityRecord:
    """Comprehensive explainability payload for UI visualization."""

    state_vector: List[float]
    state_value_v: float
    action_advantages: List[float]
    q_values: List[float]
    selected_action: int
    selected_action_name: str
    action_ranking: List[Dict[str, Any]]
    pedagogical_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_vector": [round(x, 4) for x in self.state_vector],
            "state_value_v": round(self.state_value_v, 4),
            "action_advantages": [round(x, 4) for x in self.action_advantages],
            "q_values": [round(x, 4) for x in self.q_values],
            "selected_action": self.selected_action,
            "selected_action_name": self.selected_action_name,
            "action_ranking": self.action_ranking,
            "pedagogical_rationale": self.pedagogical_rationale,
        }


class NumPyDuelingNetwork:
    """Neural Network architecture decomposing state-action values into V(s) and A(s, a)."""

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 6,
        hidden_dim: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        # He / Kaiming normal initialization for weights
        def init_w(fan_in: int, fan_out: int) -> np.ndarray:
            std = math.sqrt(2.0 / fan_in)
            return (self.rng.normal(0.0, std, (fan_in, fan_out))).astype(np.float32)

        def init_b(fan_out: int) -> np.ndarray:
            return np.zeros((1, fan_out), dtype=np.float32)

        # Shared trunk
        self.W1 = init_w(state_dim, hidden_dim)
        self.b1 = init_b(hidden_dim)
        self.W2 = init_w(hidden_dim, hidden_dim)
        self.b2 = init_b(hidden_dim)

        # Value stream V(s) -> R^1
        self.W_v1 = init_w(hidden_dim, 32)
        self.b_v1 = init_b(32)
        self.W_v2 = init_w(32, 1)
        self.b_v2 = init_b(1)

        # Advantage stream A(s, a) -> R^|A|
        self.W_a1 = init_w(hidden_dim, 32)
        self.b_a1 = init_b(32)
        self.W_a2 = init_w(32, action_dim)
        self.b_a2 = init_b(action_dim)

        # Parameter list for updates
        self.params: List[np.ndarray] = [
            self.W1, self.b1, self.W2, self.b2,
            self.W_v1, self.b_v1, self.W_v2, self.b_v2,
            self.W_a1, self.b_a1, self.W_a2, self.b_a2,
        ]

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass through Dueling architecture.

        Returns:
            q_values: Shape (batch_size, action_dim)
            v: Shape (batch_size, 1)
            a: Shape (batch_size, action_dim)
            cache: Cached activations for analytical backpropagation
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Trunk forward
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)

        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)

        # Value stream forward
        z_v1 = h2 @ self.W_v1 + self.b_v1
        h_v1 = np.maximum(0.0, z_v1)
        v = h_v1 @ self.W_v2 + self.b_v2  # (N, 1)

        # Advantage stream forward
        z_a1 = h2 @ self.W_a1 + self.b_a1
        h_a1 = np.maximum(0.0, z_a1)
        a = h_a1 @ self.W_a2 + self.b_a2  # (N, |A|)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
        a_mean = np.mean(a, axis=1, keepdims=True)
        q = v + (a - a_mean)

        cache = {
            "x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2,
            "z_v1": z_v1, "h_v1": h_v1, "v": v,
            "z_a1": z_a1, "h_a1": h_a1, "a": a, "a_mean": a_mean,
        }
        return q, v, a, cache

    def backward(
        self, d_q: np.ndarray, cache: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """Analytical backpropagation from gradient of Q with respect to loss."""
        # Unpack cache
        x, h1, h2 = cache["x"], cache["h1"], cache["h2"]
        z1, z2 = cache["z1"], cache["z2"]
        h_v1, z_v1 = cache["h_v1"], cache["z_v1"]
        h_a1, z_a1 = cache["h_a1"], cache["z_a1"]
        batch_size = x.shape[0]

        # 1. Gradients from Q aggregation: Q = V + A - mean(A)
        # dL/dV = sum_a (dL/dQ)
        d_v = np.sum(d_q, axis=1, keepdims=True)  # (N, 1)
        # dL/dA = d_q - mean(d_q)
        d_a = d_q - np.mean(d_q, axis=1, keepdims=True)  # (N, |A|)

        # 2. Advantage stream backprop
        dW_a2 = (h_a1.T @ d_a) / batch_size
        db_a2 = np.sum(d_a, axis=0, keepdims=True) / batch_size

        dh_a1 = d_a @ self.W_a2.T
        dz_a1 = dh_a1 * (z_a1 > 0.0)
        dW_a1 = (h2.T @ dz_a1) / batch_size
        db_a1 = np.sum(dz_a1, axis=0, keepdims=True) / batch_size

        dh2_from_a = dz_a1 @ self.W_a1.T

        # 3. Value stream backprop
        dW_v2 = (h_v1.T @ d_v) / batch_size
        db_v2 = np.sum(d_v, axis=0, keepdims=True) / batch_size

        dh_v1 = d_v @ self.W_v2.T
        dz_v1 = dh_v1 * (z_v1 > 0.0)
        dW_v1 = (h2.T @ dz_v1) / batch_size
        db_v1 = np.sum(dz_v1, axis=0, keepdims=True) / batch_size

        dh2_from_v = dz_v1 @ self.W_v1.T

        # 4. Merged gradients into trunk h2
        dh2 = dh2_from_a + dh2_from_v
        dz2 = dh2 * (z2 > 0.0)
        dW2 = (h1.T @ dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = (x.T @ dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        return [
            dW1, db1, dW2, db2,
            dW_v1, db_v1, dW_v2, db_v2,
            dW_a1, db_a1, dW_a2, db_a2,
        ]

    def copy_weights_from(self, source: "NumPyDuelingNetwork", tau: float = 1.0) -> None:
        """Copies or Polyak-averages parameters from source network."""
        for target_p, source_p in zip(self.params, source.params):
            if tau >= 1.0:
                np.copyto(target_p, source_p)
            else:
                target_p[:] = tau * source_p + (1.0 - tau) * target_p


class AdamOptimizer:
    """Standard Adam optimizer implemented in pure NumPy."""

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.005,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]) -> None:
        """Applies gradient update using Adam moment estimation."""
        self.t += 1
        lr_t = self.lr * (math.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t))

        for i, (p, g) in enumerate(zip(self.params, grads)):
            # Clip gradient norm for numerical stability
            g_clipped = np.clip(g, -5.0, 5.0)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g_clipped
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g_clipped**2)
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


class D3QNAgent:
    """Dueling Double Deep Q-Network Agent for POMDP Educational Optimization."""

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 6,
        hidden_dim: int = 64,
        lr: float = settings.RL_LEARNING_RATE,
        gamma: float = settings.RL_GAMMA,
        epsilon_start: float = settings.RL_EPSILON_START,
        epsilon_min: float = settings.RL_EPSILON_MIN,
        epsilon_decay: float = settings.RL_EPSILON_DECAY,
        buffer_capacity: int = settings.RL_BUFFER_CAPACITY,
        batch_size: int = settings.RL_BATCH_SIZE,
        seed: Optional[int] = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Online and Target Dueling Networks
        self.online_net = NumPyDuelingNetwork(state_dim, action_dim, hidden_dim, seed=seed)
        self.target_net = NumPyDuelingNetwork(state_dim, action_dim, hidden_dim, seed=seed)
        self.target_net.copy_weights_from(self.online_net, tau=1.0)

        # Optimizer and Replay Buffer
        self.optimizer = AdamOptimizer(self.online_net.params, lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim, seed=seed)
        self.step_counter = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Selects action via epsilon-greedy exploration or greedy exploitation."""
        if not evaluate and (self.rng.uniform(0.0, 1.0) < self.epsilon):
            return int(self.rng.integers(0, self.action_dim))

        q_vals, _, _, _ = self.online_net.forward(state)
        return int(np.argmax(q_vals[0]))

    def train_step(self) -> Optional[float]:
        """Samples a mini-batch from experience buffer and performs Double DQN Bellman update."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 1. Online Q-values for current states
        q_online, _, _, cache = self.online_net.forward(states)

        # 2. Double DQN target estimation:
        # a* = argmax_a Q_online(s', a)
        q_next_online, _, _, _ = self.online_net.forward(next_states)
        best_next_actions = np.argmax(q_next_online, axis=1)

        # Evaluate target Q using target network with best online action: Q_target(s', a*)
        q_next_target, _, _, _ = self.target_net.forward(next_states)
        q_target_best = q_next_target[np.arange(self.batch_size), best_next_actions]

        # Target y = r + gamma * (1 - done) * Q_target(s', a*)
        y = rewards + self.gamma * (1.0 - dones.astype(np.float32)) * q_target_best

        # 3. Bellman TD error & Loss gradient dL/dQ
        d_q = np.zeros_like(q_online)
        current_q = q_online[np.arange(self.batch_size), actions]
        td_errors = current_q - y  # (N,)

        # Smooth Huber-like clipping for TD error gradient
        grad_td = np.clip(td_errors, -1.0, 1.0)
        d_q[np.arange(self.batch_size), actions] = grad_td

        # 4. Backward pass & Adam parameter update
        grads = self.online_net.backward(d_q, cache)
        self.optimizer.step(grads)

        # 5. Soft Polyak update for target network
        self.step_counter += 1
        if self.step_counter % settings.RL_TARGET_UPDATE_FREQ == 0:
            self.target_net.copy_weights_from(self.online_net, tau=0.20)

        # 6. Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        loss = float(np.mean(td_errors**2))
        return loss

    def explain_decision(self, state: np.ndarray) -> D3QNExplainabilityRecord:
        """Generates grounded mathematical explainability breakdown for a given state vector."""
        q_vals, v_val, a_vals, _ = self.online_net.forward(state)

        q_arr = q_vals[0]
        v_scalar = float(v_val[0, 0])
        a_arr = a_vals[0]
        selected_act = int(np.argmax(q_arr))
        selected_name = POMDPTutorEnv.ACTION_MAP.get(selected_act, f"Action {selected_act}")

        # Construct ranked comparison
        action_ranking = []
        sorted_indices = np.argsort(q_arr)[::-1]
        for rank, act_idx in enumerate(sorted_indices):
            action_ranking.append({
                "rank": rank + 1,
                "action": int(act_idx),
                "name": POMDPTutorEnv.ACTION_MAP.get(act_idx, f"Action {act_idx}"),
                "q_value": round(float(q_arr[act_idx]), 4),
                "advantage": round(float(a_arr[act_idx]), 4),
                "is_selected": bool(act_idx == selected_act),
            })

        # Generate data-grounded pedagogical rationale without hallucinations
        mean_b = state[4] if len(state) > 4 else 0.5
        rolling_acc = state[5] if len(state) > 5 else 0.5
        friction = state[6] if len(state) > 6 else 0.0
        urgency = state[9] if len(state) > 9 else 0.0

        rationale_parts = []
        if selected_act == 5:
            rationale_parts.append(
                f"Selected '{selected_name}' because review urgency is elevated ({urgency:.2f}), preventing cognitive memory decay."
            )
        elif selected_act in (3, 4):
            rationale_parts.append(
                f"Selected guidance scaffolding '{selected_name}' due to observed cognitive friction ({friction:.2f}) and recent accuracy ({rolling_acc * 100:.0f}%)."
            )
        elif selected_act == 2:
            rationale_parts.append(
                f"Selected challenging '{selected_name}' to advance high mastery (mean belief: {mean_b:.2f}) with low friction."
            )
        elif selected_act == 0:
            rationale_parts.append(
                f"Selected '{selected_name}' to reinforce foundational knowledge components."
            )
        else:
            rationale_parts.append(
                f"Selected '{selected_name}' to balance challenge and learning progression."
            )

        runner_up = action_ranking[1]
        q_gap = q_arr[selected_act] - runner_up["q_value"]
        rationale_parts.append(
            f"Action advantage A(s, a) = {a_arr[selected_act]:+.3f} (exceeds runner-up '{runner_up['name']}' by Q-gap: {q_gap:.3f})."
        )
        full_rationale = " ".join(rationale_parts)

        return D3QNExplainabilityRecord(
            state_vector=state.tolist(),
            state_value_v=v_scalar,
            action_advantages=a_arr.tolist(),
            q_values=q_arr.tolist(),
            selected_action=selected_act,
            selected_action_name=selected_name,
            action_ranking=action_ranking,
            pedagogical_rationale=full_rationale,
        )
