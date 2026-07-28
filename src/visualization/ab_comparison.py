"""A/B Testing Split-Screen Comparison Module.

Provides dual-column side-by-side comparison for:
Visualization 13 — RL vs Traditional Tutor:
- Simultaneously steps Traditional Linear Control Tutor vs PyTorch D3QN RL Tutor.
- Displays comparative gauges and telemetry for Questions, Actions, Beliefs, Rewards, NLG, and Mastery steps.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_ab_splitscreen_view(
    rl_beliefs: List[float],
    control_beliefs: List[float],
    rl_rewards: List[float],
    control_rewards: List[float],
    rl_actions: List[str],
    control_actions: List[str],
) -> go.Figure:
    """Creates a split-screen 2-panel comparative figure benchmarking RL Tutor vs Control Tutor.

    Args:
        rl_beliefs: Trajectory of RL mean beliefs.
        control_beliefs: Trajectory of Control mean beliefs.
        rl_rewards: Trajectory of RL net rewards.
        control_rewards: Trajectory of Control net rewards.
        rl_actions: Selected action history for RL.
        control_actions: Selected action history for Control.

    Returns:
        go.Figure: Dual-panel comparison figure.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Traditional Linear Tutor (Control)</b>",
            "<b>PyTorch D3QN RL Tutor (Experimental)</b>",
        ),
        horizontal_spacing=0.12,
    )

    t_ctrl = list(range(1, len(control_beliefs) + 1))
    t_rl = list(range(1, len(rl_beliefs) + 1))

    # Left Panel: Control Tutor Belief & Reward
    if control_beliefs:
        fig.add_trace(
            go.Scatter(
                x=t_ctrl,
                y=control_beliefs,
                mode="lines+markers",
                name="Control Mean Belief",
                line=dict(color="#71717A", width=3),
                marker=dict(size=8),
                hovertemplate="Step %{x}<br>Control Belief: <b>%{y:.3f}</b><extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Right Panel: RL Tutor Belief & Reward
    if rl_beliefs:
        fig.add_trace(
            go.Scatter(
                x=t_rl,
                y=rl_beliefs,
                mode="lines+markers",
                name="RL Mean Belief",
                line=dict(color="#00E5FF", width=3.5),
                marker=dict(size=9, color="#FFFFFF"),
                hovertemplate="Step %{x}<br>RL Belief: <b>%{y:.3f}</b><extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="<b>A/B Testing Split-Screen Benchmark — Traditional Tutor vs. PyTorch D3QN Agent</b>",
            font=dict(size=16, color="#FAFAFA"),
        ),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(orientation="h", y=1.08, x=0.2),
    )

    fig.update_xaxes(gridcolor="#27272A", title="Time Step (t)")
    fig.update_yaxes(gridcolor="#27272A", title="Mean Skill Belief P(L)", range=[-0.02, 1.05])

    return fig
