"""POMDP Explainability & Student Learning Journey Module.

Provides Plotly interactive visualizations & Natural Language Explainability:
1. Visualization 6 — Hidden State vs Observable State (Synchronized side-by-side panels).
2. Visualization 7 — Student Learning Journey (Curriculum progression graph with Red/Yellow/Green status nodes).
3. Visualization 8 — RL Decision Timeline (Sequential step-by-step pipeline).
4. Visualization 12 — Explainability Card (Data-grounded Natural Language Reasoning Engine).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go
from src.curriculum import KC_NAMES


ACTION_DESCRIPTIONS: Dict[int, str] = {
    0: "Easy Practice Problem",
    1: "Medium Practice Problem",
    2: "Hard Practice Problem",
    3: "Worked Example Demonstration",
    4: "Scaffolding Hint Guidance",
}


def create_hidden_vs_observed_panel(
    obs: np.ndarray,
    beliefs: np.ndarray,
    last_correct: bool,
    last_friction: float,
    last_action_name: str,
) -> go.Figure:
    """Creates a side-by-side synchronized comparison panel for Observed vs Hidden State.

    Left Panel: Observed Telemetry (Correctness, Friction, Response Time, Action Index)
    Right Panel: Hidden Belief State (BKT P(L) per skill, Mean Belief)

    Args:
        obs: 9D observation vector.
        beliefs: 4D skill belief vector.
        last_correct: True if last answer was correct.
        last_friction: Friction score.
        last_action_name: Name of last action executed.

    Returns:
        go.Figure: Plotly 2-panel comparison figure.
    """
    fig = go.Figure()

    # Left Side: Observed Telemetry Bar Plot
    obs_labels = ["Last Correct", "Response Friction", "Step Ratio", "Last Action Norm"]
    obs_values = [
        1.0 if last_correct else 0.0,
        float(np.clip(last_friction, 0.0, 1.0)),
        float(obs[7]),
        float(obs[8]),
    ]

    fig.add_trace(
        go.Bar(
            x=obs_labels,
            y=obs_values,
            name="Observable Telemetry",
            marker=dict(color="#00E5FF", line=dict(color="#FFFFFF", width=1)),
            xaxis="x1",
            yaxis="y1",
            hovertemplate="<b>%{x}</b><br>Observed Value: %{y:.2f}<extra></extra>",
        )
    )

    # Right Side: Hidden Belief State Bar Plot
    kc_short_names = ["Basic Arith", "Adv Arith", "Basic Alg", "Adv Alg"]
    belief_values = [float(b) for b in beliefs[:4]]

    belief_colors = []
    for b in belief_values:
        if b < 0.40:
            belief_colors.append("#EF4444")
        elif b < 0.85:
            belief_colors.append("#F59E0B")
        else:
            belief_colors.append("#10B981")

    fig.add_trace(
        go.Bar(
            x=kc_short_names,
            y=belief_values,
            name="Hidden Skill Beliefs",
            marker=dict(color=belief_colors, line=dict(color="#FFFFFF", width=1)),
            xaxis="x2",
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Hidden P(L): <b>%{y:.3f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        grid=dict(rows=1, columns=2, pattern="independent"),
        title=dict(
            text="<b>POMDP Dual-Panel — Observed Info vs. Hidden Beliefs</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        xaxis1=dict(title="Observable Surface Features", domain=[0.0, 0.45]),
        yaxis1=dict(title="Observed Value", range=[0.0, 1.05], gridcolor="#27272A"),
        xaxis2=dict(title="Hidden Knowledge Components", domain=[0.55, 1.0]),
        yaxis2=dict(title="Mastery Belief P(L_t)", range=[0.0, 1.05], gridcolor="#27272A"),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=40, r=30, t=60, b=40),
        height=380,
    )

    return fig


def create_learning_journey_graph(
    beliefs: np.ndarray,
    target_kc_idx: int = 0,
    kc_names: List[str] = KC_NAMES,
) -> go.Figure:
    """Creates a curriculum progression flowchart with dynamic Red/Yellow/Green mastery nodes.

    Progression: Basic Arithmetic -> Advanced Arithmetic -> Basic Algebra -> Advanced Algebra.

    Args:
        beliefs: 4D skill belief vector.
        target_kc_idx: Index of targeted KC.
        kc_names: List of KC titles.

    Returns:
        go.Figure: Plotly graph figure.
    """
    fig = go.Figure()

    node_x = [0.12, 0.38, 0.64, 0.88]
    node_y = [0.45, 0.45, 0.45, 0.45]

    for i in range(len(node_x) - 1):
        fig.add_trace(
            go.Scatter(
                x=[node_x[i] + 0.05, node_x[i + 1] - 0.05],
                y=[0.45, 0.45],
                mode="lines+markers",
                line=dict(color="#71717A", width=3),
                marker=dict(size=10, symbol="arrow-bar-up", angle=90, color="#FFFFFF"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_colors = []
    node_borders = []
    border_widths = []
    node_texts = []

    for idx, b in enumerate(beliefs[:4]):
        kc_title = kc_names[idx] if idx < len(kc_names) else f"KC {idx}"
        b_val = float(b)

        if b_val < 0.40:
            color = "#EF4444"
            status = "Low Mastery"
        elif b_val < 0.85:
            color = "#F59E0B"
            status = "Developing"
        else:
            color = "#10B981"
            status = "Mastered"

        if idx == target_kc_idx:
            border = "#FFFFFF"
            b_width = 3.5
            status += "<br><b>[ACTIVE]</b>"
        else:
            border = "#3F3F46"
            b_width = 1.5

        node_colors.append(color)
        node_borders.append(border)
        border_widths.append(b_width)

        text_str = f"<b>{kc_title}</b><br>P(L) = <b>{b_val:.2f}</b><br>{status}"
        node_texts.append(text_str)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=[46, 46, 46, 46],
                color=node_colors,
                line=dict(color=node_borders, width=border_widths),
                symbol="square-open-dot",
            ),
            text=node_texts,
            textposition="top center",
            textfont=dict(color="#FAFAFA", size=10, family="Inter"),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Student Learning Journey — Curriculum Node Roadmap</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.02, 0.98]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.05, 0.95]),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=20, r=20, t=60, b=20),
        height=320,
    )

    return fig


def create_decision_timeline(
    current_step: int,
    action_name: str,
    target_kc_name: str,
    nlg: float,
    reward: float,
) -> go.Figure:
    """Creates a horizontal sequential step timeline for the RL execution pipeline.

    Timeline: Answer -> Obs -> BKT Update -> Belief -> D3QN Forward -> Q-values -> Action -> Reward -> Next State.

    Args:
        current_step: Current episode step.
        action_name: Name of selected action.
        target_kc_name: Name of targeted KC.
        nlg: Normalized learning gain.
        reward: Scalar net reward.

    Returns:
        go.Figure: Plotly pipeline timeline figure.
    """
    fig = go.Figure()

    stages = [
        "1. Student<br>Answer",
        "2. Surface<br>Obs (9D)",
        "3. BKT Bayes<br>Update",
        "4. Belief<br>State b_t",
        "5. D3QN<br>Forward",
        "6. Action<br>Selected",
        "7. Reward<br>NLG",
        "8. Next State<br>s_{t+1}",
    ]

    stage_x = np.linspace(0.08, 0.92, len(stages))
    stage_y = [0.45] * len(stages)

    fig.add_trace(
        go.Scatter(
            x=stage_x,
            y=stage_y,
            mode="lines",
            line=dict(color="#00E5FF", width=2.5, dash="solid"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stage_x,
            y=stage_y,
            mode="markers+text",
            marker=dict(size=32, color="#18181B", line=dict(color="#FFFFFF", width=2)),
            text=stages,
            textposition="top center",
            textfont=dict(color="#FAFAFA", size=9, family="Outfit"),
            hovertemplate="Stage: %{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>RL Sequential Pipeline — Time Step {current_step}</b>",
            font=dict(size=15, color="#FAFAFA"),
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.02, 0.98]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.05, 0.95]),
        template="plotly_dark",
        paper_bgcolor="#09090B",
        plot_bgcolor="#18181B",
        margin=dict(l=20, r=20, t=60, b=20),
        height=320,
    )

    return fig


def generate_explainability_card(
    obs: np.ndarray,
    q_values: np.ndarray,
    selected_action: int,
    val_s: float,
    adv_sa: np.ndarray,
    target_kc_name: str,
    beliefs: np.ndarray,
) -> Dict[str, Any]:
    """Generates data-grounded natural-language explainability rationale without hallucinating.

    Args:
        obs: 9D observation vector.
        q_values: Array of 5 Q-values.
        selected_action: Selected action index.
        val_s: State Value V(s).
        adv_sa: Advantage vector A(s, a).
        target_kc_name: Targeted KC name.
        beliefs: Skill belief state array.

    Returns:
        Dict[str, Any]: Dictionary containing rationale text, confidence score, expected return, and key drivers.
    """
    q_vals = np.array(q_values, dtype=np.float32)
    selected_q = float(q_vals[selected_action])

    sorted_indices = np.argsort(q_vals)[::-1]
    runner_up_action = int(sorted_indices[1]) if len(sorted_indices) > 1 else selected_action
    runner_up_q = float(q_vals[runner_up_action])
    q_margin = selected_q - runner_up_q

    target_kc_idx = int(np.argmin(beliefs[:4]))
    target_belief = float(beliefs[target_kc_idx])

    action_title = ACTION_DESCRIPTIONS.get(selected_action, f"Action {selected_action}")
    runner_up_title = ACTION_DESCRIPTIONS.get(runner_up_action, f"Action {runner_up_action}")

    exp_q = np.exp(q_vals - np.max(q_vals))
    softmax_probs = exp_q / np.sum(exp_q)
    confidence_pct = float(softmax_probs[selected_action] * 100.0)

    explanation_parts = [
        f"**Targeted Skill:** Mastery in *{target_kc_name}* is currently at **{target_belief * 100.0:.1f}%**.",
        f"**Pedagogical Rationale:** The D3QN policy selected **{action_title}** because it achieved the highest expected long-term return ($Q(s, a) = {selected_q:.2f}$).",
    ]

    if selected_action in (3, 4):
        explanation_parts.append(
            f"Because student belief is developing ({target_belief * 100.0:.0f}%), scaffolding was prioritized over raw practice to reduce cognitive friction while maximizing expected learning gain."
        )
    elif selected_action == 0:
        explanation_parts.append(
            "An Easy Practice problem was chosen to rebuild accuracy and confidence following recent response friction."
        )
    elif selected_action == 1:
        explanation_parts.append(
            f"A Medium Practice problem was selected over {runner_up_title} ($Q = {runner_up_q:.2f}$) to balance difficulty with learning gain in the Zone of Proximal Development."
        )
    elif selected_action == 2:
        explanation_parts.append(
            "A Hard Practice problem was selected to accelerate mastery towards the 90% threshold."
        )

    full_rationale = " ".join(explanation_parts)

    return {
        "rationale": full_rationale,
        "selected_action_name": action_title,
        "expected_return": selected_q,
        "confidence_pct": confidence_pct,
        "state_value_vs": float(val_s),
        "advantage_val": float(adv_sa[selected_action]),
        "q_margin": float(q_margin),
        "target_kc": target_kc_name,
        "target_belief": target_belief,
    }
