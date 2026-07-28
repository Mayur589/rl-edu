"""Dueling Double Deep Q-Network (D3QN) Decision Inspector Module.

Provides Plotly interactive visualizations for:
1. Visualization 3 — D3QN Decision Inspector (Flow diagram rendering State -> Shared Net -> V(s)/A(s,a) -> Q-values -> Action).
2. Visualization 4 — Action Comparison Panel (Horizontal Q-value bar chart highlighting best/selected/runner-up actions).
3. Visualization 5 — Reward Decomposition (Waterfall & stacked bar chart of NLG gain, step penalty, hint penalty, mastery bonus).
4. Visualization 10 — Replay Buffer Inspector (Gauge chart & experience sample table).
5. Visualization 11 — Neural Network Activity Diagram (Visual layer diagram with node activations and tensor shapes).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go
import torch

from src.pomdp_env import POMDPTutorEnv


ACTION_NAMES: List[str] = [
    "Easy Practice (Diff 0.2)",
    "Medium Practice (Diff 0.5)",
    "Hard Practice (Diff 0.8)",
    "Worked Example (Demo)",
    "Scaffolding Hint (Guide)",
]

SHORT_ACTION_NAMES: List[str] = [
    "Easy Practice",
    "Medium Practice",
    "Hard Practice",
    "Worked Example",
    "Scaffolding Hint",
]


def create_d3qn_inspector_flow(
    obs: np.ndarray,
    val_s: float,
    adv_sa: np.ndarray,
    q_values: np.ndarray,
    selected_action: int,
) -> go.Figure:
    """Creates a visual computational architecture flow diagram displaying real numerical tensors.

    Flow: State Vector (9D) -> Shared Net (128) -> Value V(s) / Advantage A(s,a) -> Q-values -> Action.

    Args:
        obs: 9D observation vector.
        val_s: Scalar State Value V(s).
        adv_sa: 5D Advantage vector A(s, a).
        q_values: 5D Q-values vector Q(s, a).
        selected_action: Selected action index.

    Returns:
        go.Figure: Plotly flow diagram figure.
    """
    fig = go.Figure()

    # Optimized node positions (x, y) with generous spacing to avoid text collisions
    node_x = [0.08, 0.26, 0.48, 0.48, 0.72, 0.92]
    node_y = [0.45, 0.45, 0.70, 0.20, 0.45, 0.45]

    node_labels = [
        f"<b>State Input s</b><br>Shape: (9,)<br>b_mean: {obs[3]:.2f}",
        "<b>Shared Trunk</b><br>FC(9→128→128)<br>ReLU",
        f"<b>State Value V(s)</b><br>Scalar: <b>{val_s:+.3f}</b><br>FC(128→64→1)",
        f"<b>Advantage A(s,a)</b><br>Dim: (5,)<br>Max: {np.max(adv_sa):+.2f}",
        f"<b>Q(s,a) Aggregator</b><br>V(s) + [A(s,a) - Ā]<br>Max Q: <b>{np.max(q_values):.2f}</b>",
        f"<b>Action Executed</b><br><b>{SHORT_ACTION_NAMES[selected_action]}</b><br>Q = {q_values[selected_action]:.2f}",
    ]

    text_positions = ["bottom center", "bottom center", "top center", "bottom center", "top center", "bottom center"]

    node_colors = [
        "#00E5FF",  # State
        "#71717A",  # Shared Trunk
        "#D500F9",  # Value Stream
        "#FFAB00",  # Advantage Stream
        "#10B981",  # Q-values
        "#FFFFFF",  # Action Selected
    ]

    # Draw connection lines
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]

    for start_idx, end_idx in edges:
        fig.add_trace(
            go.Scatter(
                x=[node_x[start_idx], node_x[end_idx]],
                y=[node_y[start_idx], node_y[end_idx]],
                mode="lines",
                line=dict(color="#3F3F46", width=2.5, dash="solid" if start_idx != 1 else "dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Draw Nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=[42, 46, 44, 44, 48, 50],
                color=node_colors,
                line=dict(color="#FFFFFF", width=2),
                symbol="square-open-dot",
            ),
            text=node_labels,
            textposition=text_positions,
            textfont=dict(color="#FAFAFA", size=10, family="Inter"),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>D3QN Decision Inspector — Live Forward Pass Tensor Flow</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.02, 1.02]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 0.95]),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=20, r=20, t=60, b=30),
        height=380,
    )

    return fig


def create_action_comparison_panel(
    q_values: np.ndarray,
    selected_action: int,
) -> go.Figure:
    """Creates a horizontal bar chart comparing Q-values for all 5 discrete actions.

    Highlights:
        - Highest Q-value (Green)
        - Selected Action (White outline / Active marker)
        - Second best action (Subtle gray)

    Args:
        q_values: Array of 5 Q-values.
        selected_action: Action index chosen by policy.

    Returns:
        go.Figure: Plotly horizontal bar chart figure.
    """
    fig = go.Figure()

    q_vals = np.array(q_values, dtype=np.float32)
    best_action = int(np.argmax(q_vals))

    sorted_indices = np.argsort(q_vals)[::-1]
    second_best_action = int(sorted_indices[1]) if len(sorted_indices) > 1 else best_action

    colors = []
    borders = []
    border_widths = []

    for i in range(len(q_vals)):
        if i == selected_action:
            borders.append("#FFFFFF")
            border_widths.append(3)
        else:
            borders.append("#3F3F46")
            border_widths.append(1)

        if i == best_action:
            colors.append("#10B981")  # Emerald Green
        elif i == second_best_action:
            colors.append("#3B82F6")  # Blue
        else:
            colors.append("#27272A")

    bar_texts = []
    for i, q in enumerate(q_vals):
        tag = ""
        if i == selected_action:
            tag += " [SELECTED]"
        if i == best_action:
            tag += " (BEST)"
        bar_texts.append(f"Q = {q:.3f}{tag}")

    fig.add_trace(
        go.Bar(
            y=SHORT_ACTION_NAMES,
            x=q_vals,
            orientation="h",
            text=bar_texts,
            textposition="inside",
            textfont=dict(color="#FFFFFF", size=11, family="Outfit"),
            marker=dict(
                color=colors,
                line=dict(color=borders, width=border_widths),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Q-value: <b>%{x:.3f}</b><br>"
                + "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Action Comparison Panel — Q(s, a) State-Action Values</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(
            title="Expected Return Q(s, a)",
            gridcolor="#27272A",
            zerolinecolor="#3F3F46",
        ),
        yaxis=dict(
            autorange="reversed",
        ),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=130, r=30, t=60, b=40),
        height=380,
    )

    return fig


def create_reward_decomposition_chart(
    nlg: float,
    is_hint_spam: bool = False,
    is_mastery: bool = False,
) -> go.Figure:
    """Creates a waterfall chart breaking down scalar reward contributions.

    Formula:
        + Learning Gain (5.0 * NLG)
        - Step Penalty (-0.50)
        - Hint Spam Penalty (-1.00 if > 2 hints)
        + Mastery Bonus (+50.0 if mean belief >= 0.90)
        = Final Net Reward

    Args:
        nlg: Normalized Learning Gain scalar.
        is_hint_spam: True if consecutive hints > 2.
        is_mastery: True if mean belief >= 0.90.

    Returns:
        go.Figure: Plotly waterfall chart.
    """
    fig = go.Figure()

    reward_nlg = float(5.0 * nlg)
    penalty_step = -0.50
    penalty_hint = -1.00 if is_hint_spam else 0.0
    bonus_mastery = 50.0 if is_mastery else 0.0

    total_reward = reward_nlg + penalty_step + penalty_hint + bonus_mastery

    x_labels = [
        "NLG Gain (5x)",
        "Step Penalty",
        "Hint Penalty",
        "Mastery Bonus",
        "Total Reward",
    ]

    measures = ["relative", "relative", "relative", "relative", "total"]
    values = [reward_nlg, penalty_step, penalty_hint, bonus_mastery, 0]

    fig.add_trace(
        go.Waterfall(
            name="Reward Decomposition",
            orientation="v",
            measure=measures,
            x=x_labels,
            y=values,
            text=[f"{v:+.2f}" if m == "relative" else f"{total_reward:+.2f}" for v, m in zip(values, measures)],
            textposition="outside",
            textfont=dict(color="#FAFAFA", size=11, family="Outfit"),
            connector=dict(line=dict(color="#3F3F46", width=2)),
            increasing=dict(marker=dict(color="#10B981")),
            decreasing=dict(marker=dict(color="#EF4444")),
            totals=dict(marker=dict(color="#FFFFFF", line=dict(color="#FFFFFF", width=2))),
            hovertemplate="<b>%{x}</b><br>Contribution: %{y:+.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Multi-Objective Reward Decomposition</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        yaxis=dict(
            title="Reward Units",
            gridcolor="#27272A",
            zerolinecolor="#3F3F46",
        ),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=40, r=30, t=60, b=40),
        height=380,
    )

    return fig


def create_replay_buffer_inspector(
    capacity: int = 50000,
    current_size: int = 0,
    recent_transitions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[go.Figure, List[Dict[str, Any]]]:
    """Creates a gauge chart for replay buffer capacity and lists sampled transitions.

    Args:
        capacity: Maximum buffer size.
        current_size: Current stored transition count.
        recent_transitions: Optional list of recent stored transition dicts.

    Returns:
        Tuple[go.Figure, List[Dict[str, Any]]]: Gauge figure and transition list.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_size,
            title=dict(text="<b>Replay Buffer Memory Capacity</b>", font=dict(size=14, color="#FAFAFA")),
            delta=dict(reference=capacity, relative=False, valueformat="d"),
            gauge=dict(
                axis=dict(range=[0, capacity], tickwidth=1, tickcolor="#71717A"),
                bar=dict(color="#00E5FF"),
                bgcolor="#18181B",
                borderwidth=2,
                bordercolor="#27272A",
                steps=[
                    {"range": [0, capacity * 0.5], "color": "#121215"},
                    {"range": [capacity * 0.5, capacity * 0.85], "color": "#18181B"},
                    {"range": [capacity * 0.85, capacity], "color": "#27272A"},
                ],
                threshold=dict(
                    line=dict(color="#10B981", width=4),
                    thickness=0.75,
                    value=capacity * 0.8,
                ),
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=30, r=30, t=40, b=20),
        height=220,
    )

    transitions = recent_transitions if recent_transitions else []
    return fig, transitions


def create_neural_activity_diagram(
    state_dim: int = 9,
    hidden_dim: int = 128,
    action_dim: int = 5,
    last_obs: Optional[np.ndarray] = None,
) -> go.Figure:
    """Creates a visual layer diagram illustrating the Dueling DQN network architecture and activations.

    Layers:
        Input State (9) -> FC Hidden 1 (128) -> FC Hidden 2 (128) -> [V(s) (1) | A(s,a) (5)] -> Q(s,a) (5)

    Args:
        state_dim: Input vector size (9).
        hidden_dim: Hidden dimension size (128).
        action_dim: Number of actions (5).
        last_obs: Optional observation array.

    Returns:
        go.Figure: Plotly architecture layer figure.
    """
    fig = go.Figure()

    layers = ["Input State<br>(9D)", "Hidden Layer 1<br>(128D)", "Hidden Layer 2<br>(128D)", "Streams<br>V(1) / A(5)", "Q-Values<br>(5D)"]
    nodes_per_layer = [9, 12, 12, 6, 5]
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    for i in range(len(layers) - 1):
        x1, x2 = x_positions[i], x_positions[i + 1]
        n1, n2 = nodes_per_layer[i], nodes_per_layer[i + 1]

        y1_list = np.linspace(0.12, 0.88, n1)
        y2_list = np.linspace(0.12, 0.88, n2)

        for y1 in y1_list:
            for y2 in y2_list:
                fig.add_trace(
                    go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode="lines",
                        line=dict(color="rgba(63, 63, 70, 0.20)", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    for i, (layer_name, n_nodes, x_pos) in enumerate(zip(layers, nodes_per_layer, x_positions)):
        y_list = np.linspace(0.12, 0.88, n_nodes)
        node_color = "#00E5FF" if i == 0 else "#D500F9" if i == 3 else "#10B981" if i == 4 else "#FFFFFF"

        fig.add_trace(
            go.Scatter(
                x=[x_pos] * n_nodes,
                y=y_list,
                mode="markers+text",
                marker=dict(size=12, color=node_color, line=dict(color="#FFFFFF", width=1.5)),
                text=[layer_name] + [""] * (n_nodes - 1),
                textposition="top center",
                textfont=dict(color="#FAFAFA", size=9, family="Inter"),
                hoverinfo="text",
                hovertext=[f"{layer_name} - Subsampled Node {idx+1}" for idx in range(n_nodes)],
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Dueling DQN Layer Architecture & Feature Forward Pass</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.02, 0.98]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.0, 1.0]),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=20, r=20, t=60, b=20),
        height=380,
    )

    return fig
