"""System Architecture Animation & Global Session Analytics Module.

Provides Plotly interactive visualizations for:
1. Visualization 14 — Session Analytics Dashboard (Historical interaction analytics, belief variance, response time, action frequency).
2. Visualization 15 — Full Architecture Animation (Interactive end-to-end flow diagram highlighting active components).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_full_architecture_diagram(active_step_idx: Optional[int] = None) -> go.Figure:
    """Creates an end-to-end system architecture animation diagram.

    Flow: Student -> Environment -> Obs -> BKT Engine -> Belief Vector -> State Vector -> D3QN -> Q-values -> Action Selection -> Teaching Action -> Student

    Args:
        active_step_idx: Optional step index to animate active flow node.

    Returns:
        go.Figure: System architecture flowchart figure.
    """
    fig = go.Figure()

    nodes = [
        "1. Human Student<br>Cognitive Response",
        "2. Gymnasium Env<br>(POMDPTutorEnv)",
        "3. Surface Obs<br>(Accuracy, Friction)",
        "4. BKT Engine<br>(Bayes P(L) Updates)",
        "5. Belief Vector b_t<br>(4-Skill Probabilities)",
        "6. State Vector s<br>(9D Feature Tensor)",
        "7. Dueling DQN<br>V(s) + A(s,a)",
        "8. Q-values Q(s,a)<br>(5 Action Values)",
        "9. Action Selection<br>argmax Q(s,a)",
        "10. Teaching Action<br>(Diff / Hint / Demo)",
    ]

    # Two rows layout (top row: 1-6, bottom row: 7-10)
    node_x = [0.08, 0.24, 0.40, 0.56, 0.72, 0.88, 0.88, 0.60, 0.36, 0.12]
    node_y = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.20, 0.20, 0.20, 0.20]

    num_nodes = len(nodes)
    for i in range(num_nodes):
        next_i = (i + 1) % num_nodes
        x1, y1 = node_x[i], node_y[i]
        x2, y2 = node_x[next_i], node_y[next_i]

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color="#3F3F46", width=2.5, dash="solid" if i < 9 else "dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    colors = []
    borders = []
    sizes = []

    for i in range(num_nodes):
        if active_step_idx is not None and (i == (active_step_idx % num_nodes)):
            colors.append("#00E5FF")
            borders.append("#FFFFFF")
            sizes.append(44)
        else:
            colors.append("#18181B")
            borders.append("#71717A")
            sizes.append(36)

    text_positions = ["top center"] * 6 + ["bottom center"] * 4

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color=borders, width=2),
                symbol="square-open-dot",
            ),
            text=nodes,
            textposition=text_positions,
            textfont=dict(color="#FAFAFA", size=9, family="Outfit"),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Full Intelligent Tutoring System Architecture & Data Flow Loop</b>",
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
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
    )

    return fig


def create_session_analytics_dashboard(
    session_df: pd.DataFrame,
    survey_df: pd.DataFrame,
) -> go.Figure:
    """Creates a 4-panel global session research analytics figure.

    Panels:
        1. Learning Gain Distribution (NLG Boxplot by Group)
        2. Steps to Mastery Efficiency Comparison
        3. Pedagogical Action Frequency Breakdown
        4. Response Time vs. Belief Correlation Scatter Plot

    Args:
        session_df: Processed interaction session DataFrame.
        survey_df: Affective survey responses DataFrame.

    Returns:
        go.Figure: 4-panel analytics figure.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>1. Normalized Learning Gain (NLG) Distribution</b>",
            "<b>2. Steps to Mastery (Time-on-Task)</b>",
            "<b>3. Action Choice Selection Frequency (%)</b>",
            "<b>4. Response Time vs. Belief Correlation</b>",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    if session_df.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#09090B",
            plot_bgcolor="#18181B",
            title="Global Telemetry Analytics Dashboard (No interaction data available)",
        )
        return fig

    if "mode_label" not in session_df.columns and "tutor_mode" in session_df.columns:
        session_df["mode_label"] = session_df["tutor_mode"].apply(
            lambda x: "RL Mode" if ("Experimental" in str(x) or "RL" in str(x)) else "Control Mode"
        )
    elif "mode_label" not in session_df.columns:
        session_df["mode_label"] = "RL Mode"

    if "mean_belief" in session_df.columns:
        gains = session_df.groupby(["student_id", "mode_label"])["mean_belief"].agg(["first", "last"]).reset_index()
        denom = np.maximum(1e-4, 1.0 - gains["first"])
        gains["nlg"] = (gains["last"] - gains["first"]) / denom

        fig_nlg = px.box(
            gains,
            x="mode_label",
            y="nlg",
            color="mode_label",
            color_discrete_map={"RL Mode": "#00E5FF", "Control Mode": "#71717A"},
        )
        for trace in fig_nlg.data:
            fig.add_trace(trace, row=1, col=1)

    if "step" in session_df.columns:
        steps_df = session_df.groupby(["student_id", "mode_label"])["step"].max().reset_index()
        fig_steps = px.box(
            steps_df,
            x="mode_label",
            y="step",
            color="mode_label",
            color_discrete_map={"RL Mode": "#00E5FF", "Control Mode": "#71717A"},
        )
        for trace in fig_steps.data:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

    if "action_name" in session_df.columns:
        act_counts = session_df.groupby(["mode_label", "action_name"]).size().unstack(fill_value=0)
        act_pcts = act_counts.div(act_counts.sum(axis=1), axis=0) * 100.0

        for col_name in act_pcts.columns:
            fig.add_trace(
                go.Bar(
                    x=act_pcts.index,
                    y=act_pcts[col_name],
                    name=str(col_name),
                ),
                row=2,
                col=1,
            )

    if "response_time_sec" in session_df.columns and "mean_belief" in session_df.columns:
        fig.add_trace(
            go.Scatter(
                x=session_df["mean_belief"],
                y=session_df["response_time_sec"],
                mode="markers",
                marker=dict(color="#D500F9", size=6, opacity=0.6),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="<b>Supervisor Analytics Dashboard — Global Telemetry & Statistical Efficacy</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=40, r=30, t=70, b=40),
        legend=dict(orientation="h", y=1.05, x=0.1),
    )

    fig.update_xaxes(gridcolor="#27272A")
    fig.update_yaxes(gridcolor="#27272A")

    return fig
