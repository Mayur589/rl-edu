"""Bayesian Knowledge Tracing (BKT) Visualization Module.

Provides Plotly interactive charts for:
1. Live BKT Belief Evolution line chart with confidence bands and summary metrics.
2. Interactive Belief Heatmap tracking KC mastery over time steps.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go
from src.curriculum import KC_NAMES


# High-contrast research color palette per KC
KC_COLORS: Dict[int, str] = {
    0: "#00E5FF",  # Basic Arithmetic - Electric Cyan
    1: "#D500F9",  # Advanced Arithmetic - Vivid Magenta
    2: "#FFAB00",  # Basic Algebra - Warm Amber
    3: "#00E676",  # Advanced Algebra - Bright Emerald
}


def create_bkt_evolution_chart(
    history_beliefs: List[List[float]],
    active_kc_idx: Optional[int] = None,
    kc_names: List[str] = KC_NAMES,
) -> Tuple[go.Figure, Dict[str, float]]:
    """Creates an animated, research-grade Plotly line chart tracking BKT belief evolution.

    Args:
        history_beliefs: List of shape (T, num_skills) containing belief vectors over time steps.
        active_kc_idx: Optional index of the currently targeted KC to highlight.
        kc_names: List of KC titles.

    Returns:
        Tuple[go.Figure, Dict[str, float]]: Plotly figure object and metric summary dictionary.
    """
    fig = go.Figure()

    if not history_beliefs:
        # Fallback empty plot state
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#09090B",
            plot_bgcolor="#18181B",
            title="BKT Belief Trajectory (Waiting for session...)",
        )
        return fig, {"mean_belief": 0.0, "learning_gain": 0.0, "mastery_ratio": 0.0}

    arr_beliefs = np.array(history_beliefs, dtype=np.float32)  # Shape (T, K)
    num_steps, num_kcs = arr_beliefs.shape
    steps = list(range(1, num_steps + 1))

    # Add lines and confidence bands for each Knowledge Component
    for kc_idx in range(num_kcs):
        kc_name = kc_names[kc_idx] if kc_idx < len(kc_names) else f"KC {kc_idx}"
        color = KC_COLORS.get(kc_idx, "#FFFFFF")
        values = arr_beliefs[:, kc_idx]

        # Calculate standard error / confidence interval band
        # SE = sqrt(b * (1 - b) / max(1, step))
        se = np.sqrt(np.clip(values * (1.0 - values), 1e-4, 1.0) / np.maximum(1, np.array(steps)))
        upper_band = np.clip(values + 1.96 * se * 0.3, 0.0, 1.0)
        lower_band = np.clip(values - 1.96 * se * 0.3, 0.0, 1.0)

        # Upper bound (invisible line for fill)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=upper_band,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound with translucent shaded fill
        # Convert hex color to rgba
        hex_color = color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        fill_color = f"rgba({r}, {g}, {b}, 0.12)"

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=lower_band,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Main trajectory line
        is_active = (active_kc_idx is not None) and (kc_idx == active_kc_idx)
        line_width = 4 if is_active else 2.5
        opacity = 1.0 if (active_kc_idx is None or is_active) else 0.55

        marker_size = [8 if t < num_steps else 14 for t in range(num_steps)]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines+markers",
                name=kc_name + (" (Active Target)" if is_active else ""),
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size, symbol="circle"),
                opacity=opacity,
                hovertemplate=(
                    f"<b>{kc_name}</b><br>"
                    + "Time Step: %{x}<br>"
                    + "Mastery P(L): <b>%{y:.3f}</b><br>"
                    + "<extra></extra>"
                ),
            )
        )

    # Calculate Summary Metrics
    current_beliefs = arr_beliefs[-1]
    initial_beliefs = arr_beliefs[0]

    mean_belief = float(np.mean(current_beliefs))
    initial_mean = float(np.mean(initial_beliefs))

    denom = max(1e-4, 1.0 - initial_mean)
    learning_gain = float((mean_belief - initial_mean) / denom)
    mastery_ratio = float(np.sum(current_beliefs >= 0.85) / float(num_kcs))

    metrics = {
        "mean_belief": mean_belief,
        "learning_gain": learning_gain,
        "mastery_ratio": mastery_ratio,
    }

    # Chart Layout
    fig.update_layout(
        title=dict(
            text="<b>Live BKT Belief Trajectory (P(L) Mastery Evolution)</b>",
            font=dict(size=16, color="#FAFAFA"),
        ),
        xaxis=dict(
            title="Environment Time Step (t)",
            gridcolor="#27272A",
            zerolinecolor="#3F3F46",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(
            title="Mastery Probability P(L_t)",
            range=[-0.02, 1.05],
            gridcolor="#27272A",
            zerolinecolor="#3F3F46",
            tickformat=".2f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#E4E4E7"),
            bgcolor="rgba(24, 24, 27, 0.7)",
        ),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=50, r=30, t=60, b=50),
        hovermode="x unified",
    )

    return fig, metrics


def create_bkt_heatmap(
    history_beliefs: List[List[float]],
    kc_names: List[str] = KC_NAMES,
) -> go.Figure:
    """Creates an interactive Plotly Heatmap showing skill mastery over time steps.

    Args:
        history_beliefs: List of shape (T, num_skills) containing belief vectors over time steps.
        kc_names: List of KC titles.

    Returns:
        go.Figure: Plotly heatmap figure.
    """
    fig = go.Figure()

    if not history_beliefs:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#09090B",
            plot_bgcolor="#18181B",
            title="BKT Mastery Heatmap (Waiting for interaction steps...)",
        )
        return fig

    arr = np.array(history_beliefs, dtype=np.float32)  # (T, K)
    arr_t = arr.T  # Transpose to (K, T) so Rows = Skills, Cols = Time steps

    num_kcs, num_steps = arr_t.shape
    x_steps = [f"Step {t+1}" for t in range(num_steps)]
    y_kcs = [kc_names[i] if i < len(kc_names) else f"KC {i}" for i in range(num_kcs)]

    # Custom Red -> Yellow -> Green Color Scale
    custom_colorscale = [
        [0.0, "#EF4444"],   # Bright Red (Low mastery)
        [0.35, "#F59E0B"],  # Amber Yellow (Developing mastery)
        [0.70, "#10B981"],  # Emerald Green (High mastery)
        [1.0, "#065F46"],   # Deep Forest Green (Full mastery)
    ]

    # Create text annotations matrix for cells
    text_matrix = [[f"{val:.2f}" for val in row] for row in arr_t]

    fig.add_trace(
        go.Heatmap(
            z=arr_t,
            x=x_steps,
            y=y_kcs,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(color="#FFFFFF", size=12, family="Outfit"),
            colorscale=custom_colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title="P(L)",
                ticks="outside",
                tickformat=".1f",
                thickness=15,
                len=0.9,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "%{x}<br>"
                + "Mastery Probability: <b>%{z:.3f}</b><br>"
                + "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Interactive BKT Mastery Heatmap Across Time</b>",
            font=dict(size=16, color="#FAFAFA"),
        ),
        xaxis=dict(
            title="Time Step Progression",
            gridcolor="#27272A",
        ),
        yaxis=dict(
            title="Knowledge Component",
            autorange="reversed",  # KC 0 on top
        ),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=120, r=30, t=60, b=50),
    )

    return fig
