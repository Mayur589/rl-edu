"""Explainable RL Visualization Package for POMDP-BKT Intelligent Tutoring System.

This package provides modular, research-grade interactive Plotly visualization,
D3QN decision inspection, BKT belief trajectory tracking, natural language explainability,
and system architecture animation tools.
"""

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

__all__ = [
    "create_bkt_evolution_chart",
    "create_bkt_heatmap",
    "create_d3qn_inspector_flow",
    "create_action_comparison_panel",
    "create_reward_decomposition_chart",
    "create_replay_buffer_inspector",
    "create_neural_activity_diagram",
    "create_hidden_vs_observed_panel",
    "create_learning_journey_graph",
    "create_decision_timeline",
    "generate_explainability_card",
    "create_training_policy_dashboard",
    "create_ab_splitscreen_view",
    "create_session_analytics_dashboard",
    "create_full_architecture_diagram",
]
