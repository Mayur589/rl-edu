"""Policy Evolution Training Monitor Visualization Module.

Provides Plotly interactive dashboard monitoring D3QN policy evolution during training:
1. Visualization 9 — Policy Evolution During Training:
   - Episode Reward & Moving Average.
   - Exploration Rate (Epsilon decay).
   - Action Distribution Breakdown over episodes.
   - Training Loss Curve (Bellman residual).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SHORT_ACTION_NAMES: List[str] = [
    "Easy Practice",
    "Medium Practice",
    "Hard Practice",
    "Worked Example",
    "Scaffolding Hint",
]


def create_training_policy_dashboard(
    episode_rewards: List[float],
    episode_losses: List[float],
    epsilons: List[float],
    action_counts_history: List[List[int]],
    window_size: int = 50,
) -> go.Figure:
    """Creates a large, high-resolution 4-panel Plotly figure displaying policy evolution during training.

    Args:
        episode_rewards: List of cumulative rewards per episode.
        episode_losses: List of mean training loss per episode.
        epsilons: List of exploration rate values per episode.
        action_counts_history: Matrix of shape (episodes, 5) tracking action frequencies.
        window_size: Rolling average window size.

    Returns:
        go.Figure: 4-panel subplot figure.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>1. Cumulative Episode Reward & 50-Ep Moving Avg</b>",
            "<b>2. Exploration Rate (Epsilon Decay ε)</b>",
            "<b>3. Action Choice Usage Distribution (%)</b>",
            "<b>4. Training Loss Curve (Bellman TD Loss)</b>",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    if not episode_rewards:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#09090B",
            plot_bgcolor="#18181B",
            title="D3QN Policy Training Telemetry Dashboard (Awaiting training run...)",
            height=700,
        )
        return fig

    episodes = list(range(1, len(episode_rewards) + 1))

    # Panel 1: Episode Reward & Moving Average
    rewards_arr = np.array(episode_rewards, dtype=np.float32)
    moving_avg = np.convolve(rewards_arr, np.ones(window_size) / window_size, mode="valid")
    ma_episodes = list(range(window_size, len(episode_rewards) + 1))

    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=rewards_arr,
            mode="lines",
            name="Episode Reward",
            line=dict(color="rgba(0, 229, 255, 0.25)", width=1.5),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    if len(moving_avg) > 0:
        fig.add_trace(
            go.Scatter(
                x=ma_episodes,
                y=moving_avg,
                mode="lines",
                name=f"{window_size}-Ep Moving Avg",
                line=dict(color="#00E5FF", width=3.5),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Epsilon Decay
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=epsilons,
            mode="lines",
            name="Epsilon Rate ε",
            line=dict(color="#FFAB00", width=3),
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Panel 3: Action Distribution History
    if action_counts_history:
        act_mat = np.array(action_counts_history, dtype=np.float32)  # (episodes, 5)
        total_acts = np.maximum(1.0, np.sum(act_mat, axis=1, keepdims=True))
        act_pcts = (act_mat / total_acts) * 100.0

        colors = ["#00E5FF", "#D500F9", "#FFAB00", "#10B981", "#FFFFFF"]
        for act_idx in range(5):
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=act_pcts[:, act_idx],
                    mode="lines",
                    name=SHORT_ACTION_NAMES[act_idx],
                    line=dict(color=colors[act_idx], width=2),
                    stackgroup="one",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

    # Panel 4: Loss Curve
    if episode_losses:
        losses_arr = np.array(episode_losses, dtype=np.float32)
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=losses_arr,
                mode="lines",
                name="TD Loss",
                line=dict(color="#EF4444", width=2.5),
                showlegend=True,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="<b>PyTorch Dueling Double DQN — Policy Training Dashboard</b>",
            font=dict(size=18, color="#FAFAFA"),
            y=0.98,
            x=0.01,
            xanchor="left",
            yanchor="top",
        ),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=60, r=40, t=75, b=60),
        height=720,
        legend=dict(orientation="h", y=1.03, x=0.05, font=dict(size=11)),
    )

    fig.update_xaxes(gridcolor="#27272A", title="Training Episodes", title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#27272A", title_font=dict(size=12))

    return fig
