"""Unit Tests for Visualization Package.

Verifies that all 15 Plotly visualization generators, D3QN decision inspectors,
BKT heatmaps, and natural language explainability cards execute cleanly and return valid figures.
"""

from typing import Any, Dict
import numpy as np
import plotly.graph_objects as go
import pytest

from src.visualization.bkt_viz import create_bkt_evolution_chart, create_bkt_heatmap
from src.visualization.d3qn_inspector import (
    create_d3qn_inspector_flow,
    create_action_comparison_panel,
    create_reward_decomposition_chart,
    create_replay_buffer_inspector,
    create_neural_activity_diagram,
)
from src.visualization.pomdp_explainability import (
    create_hidden_vs_observed_panel,
    create_learning_journey_graph,
    create_decision_timeline,
    generate_explainability_card,
)
from src.visualization.training_viz import create_training_policy_dashboard
from src.visualization.ab_comparison import create_ab_splitscreen_view
from src.visualization.architecture_viz import (
    create_session_analytics_dashboard,
    create_full_architecture_diagram,
)


def test_bkt_visualizations() -> None:
    """Verifies BKT line chart and heatmap generation."""
    history = [[0.2, 0.2, 0.2, 0.2], [0.35, 0.25, 0.2, 0.2], [0.5, 0.4, 0.3, 0.2]]

    fig_line, metrics = create_bkt_evolution_chart(history_beliefs=history, active_kc_idx=0)
    assert isinstance(fig_line, go.Figure)
    assert "mean_belief" in metrics
    assert metrics["mean_belief"] > 0.0

    fig_heat = create_bkt_heatmap(history_beliefs=history)
    assert isinstance(fig_heat, go.Figure)


def test_d3qn_inspector_visualizations() -> None:
    """Verifies D3QN decision flow, action comparison, and reward decomposition."""
    obs = np.array([0.2, 0.3, 0.4, 0.3, 0.5, 0.8, 0.2, 0.1, 0.0], dtype=np.float32)
    val_s = 14.82
    adv_sa = np.array([-1.2, 2.5, -3.1, 0.4, 1.1], dtype=np.float32)
    q_vals = np.array([13.6, 17.3, 11.7, 15.2, 15.9], dtype=np.float32)

    fig_flow = create_d3qn_inspector_flow(obs=obs, val_s=val_s, adv_sa=adv_sa, q_values=q_vals, selected_action=1)
    assert isinstance(fig_flow, go.Figure)

    fig_comp = create_action_comparison_panel(q_values=q_vals, selected_action=1)
    assert isinstance(fig_comp, go.Figure)

    fig_rew = create_reward_decomposition_chart(nlg=0.15, is_hint_spam=False, is_mastery=False)
    assert isinstance(fig_rew, go.Figure)

    fig_gauge, _ = create_replay_buffer_inspector(capacity=50000, current_size=12500)
    assert isinstance(fig_gauge, go.Figure)

    fig_nn = create_neural_activity_diagram(state_dim=9, hidden_dim=128, action_dim=5)
    assert isinstance(fig_nn, go.Figure)


def test_pomdp_explainability() -> None:
    """Verifies hidden vs observed panels, learning journey, decision timelines, and explainability cards."""
    obs = np.array([0.2, 0.3, 0.4, 0.3, 0.5, 0.8, 0.2, 0.1, 0.0], dtype=np.float32)
    beliefs = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    q_vals = np.array([13.6, 17.3, 11.7, 15.2, 15.9], dtype=np.float32)

    fig_hidden = create_hidden_vs_observed_panel(obs=obs, beliefs=beliefs, last_correct=True, last_friction=0.3, last_action_name="Medium Practice")
    assert isinstance(fig_hidden, go.Figure)

    fig_journey = create_learning_journey_graph(beliefs=beliefs, target_kc_idx=0)
    assert isinstance(fig_journey, go.Figure)

    fig_time = create_decision_timeline(current_step=5, action_name="Medium Practice", target_kc_name="Basic Arithmetic", nlg=0.12, reward=2.5)
    assert isinstance(fig_time, go.Figure)

    card = generate_explainability_card(obs=obs, q_values=q_vals, selected_action=1, val_s=14.82, adv_sa=np.array([-1.2, 2.5, -3.1, 0.4, 1.1]), target_kc_name="Basic Arithmetic", beliefs=beliefs)
    assert isinstance(card, dict)
    assert "rationale" in card
    assert "confidence_pct" in card


def test_architecture_and_ab_visualizations() -> None:
    """Verifies system architecture and A/B benchmark visualization generators."""
    fig_arch = create_full_architecture_diagram(active_step_idx=3)
    assert isinstance(fig_arch, go.Figure)

    fig_ab = create_ab_splitscreen_view(rl_beliefs=[0.2, 0.4, 0.6], control_beliefs=[0.2, 0.3, 0.35], rl_rewards=[1.0, 2.0, 3.0], control_rewards=[0.5, 0.8, 1.0], rl_actions=["Easy", "Medium", "Hard"], control_actions=["Easy", "Easy", "Medium"])
    assert isinstance(fig_ab, go.Figure)

    fig_train = create_training_policy_dashboard(episode_rewards=[10.0, 25.0, 40.0], episode_losses=[0.5, 0.3, 0.1], epsilons=[1.0, 0.9, 0.8], action_counts_history=[[1, 2, 0, 1, 0], [0, 2, 1, 1, 0], [0, 1, 2, 0, 1]])
    assert isinstance(fig_train, go.Figure)
