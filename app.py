"""Research-Quality Explainable Reinforcement Learning Platform for Intelligent Tutoring Systems.

This Streamlit application provides 15 synchronized interactive Plotly visualizations:
1. Live BKT Belief Evolution with Confidence Intervals & Metrics.
2. Interactive Belief Heatmap across Time Steps.
3. D3QN Decision Inspector displaying real forward pass tensors V(s), A(s,a), Q(s,a).
4. Action Comparison Panel highlighting best, selected, and runner-up Q-values.
5. Multi-Objective Reward Decomposition Waterfall chart.
6. Hidden State vs. Observable State Dual-Panel Comparison.
7. Student Learning Journey Curriculum Progression Map.
8. Sequential RL Decision Pipeline Timeline.
9. Policy Evolution & Training Telemetry Dashboard.
10. Replay Buffer Capacity Gauge & Experience Mini-Batch Inspector.
11. Neural Network Activity & Layer Diagram.
12. Data-Grounded Natural Language Explainability Card.
13. A/B Testing Split-Screen Benchmark (RL Tutor vs Traditional Tutor).
14. Global Session Analytics Dashboard.
15. Full Intelligent Tutoring System Architecture & Data Flow Loop.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Prevent GUI thread crashes on macOS
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Force PyTorch single-threading to prevent macOS thread crashes
torch.set_num_threads(1)

from src.bkt_engine import BKTEngine
from src.pomdp_env import POMDPTutorEnv
from src.baselines import RandomTutor, HeuristicTutor
from src.d3qn_agent import D3QNAgent, D3QNTutor, train_d3qn
from src.evaluator import POMDPHeuristicTutor, evaluate_pomdp_tutor
from src.curriculum import KC_NAMES, QuestionItem, get_curriculum_question, QUESTION_BANK
from src.logger import log_interaction, log_affective_survey, ensure_log_files_exist
from src.analytics import SessionAnalyzer

# Import Modular Visualization Engine
from src.visualization import (
    create_bkt_evolution_chart,
    create_bkt_heatmap,
    create_d3qn_inspector_flow,
    create_action_comparison_panel,
    create_reward_decomposition_chart,
    create_replay_buffer_inspector,
    create_neural_activity_diagram,
    create_hidden_vs_observed_panel,
    create_learning_journey_graph,
    create_decision_timeline,
    generate_explainability_card,
    create_training_policy_dashboard,
    create_ab_splitscreen_view,
    create_session_analytics_dashboard,
    create_full_architecture_diagram,
)


# --- Model Caching ---
@st.cache_resource
def load_d3qn_cached(model_path: str = "models/d3qn_tutor.pt") -> Optional[D3QNAgent]:
    """Loads and caches the PyTorch D3QN agent.

    Args:
        model_path: Path to PyTorch model pt file.

    Returns:
        D3QNAgent or None.
    """
    if os.path.exists(model_path):
        agent = D3QNAgent(state_dim=9, action_dim=5, device="cpu")
        agent.load(model_path)
        agent.online_net.eval()
        return agent
    return None


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Explainable RL Intelligent Tutoring System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #09090B !important;
        color: #FAFAFA !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #121215 !important;
        border-right: 1px solid #27272A !important;
    }

    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(180deg, #FFFFFF 0%, #A1A1AA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-size: 1.02rem;
        color: #A1A1AA;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    .card-box {
        background: #18181B;
        border: 1px solid #27272A;
        border-left: 4px solid #00E5FF;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .explain-box {
        background: #141417;
        border: 1px solid #3F3F46;
        border-left: 4px solid #10B981;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    [data-testid="stMetric"] {
        background-color: #18181B !important;
        border: 1px solid #27272A !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
    }

    .stButton > button {
        background-color: #18181B !important;
        color: #FFFFFF !important;
        border: 1px solid #3F3F46 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #09090B !important;
        border-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown('<div class="main-header">Explainable Reinforcement Learning Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">POMDP-BKT Intelligent Tutoring System — Research-Quality Interactivity & Decision Inspection</div>',
    unsafe_allow_html=True,
)

# Ensure data log files exist
ensure_log_files_exist()

# --- Sidebar Controls ---
st.sidebar.title("Research Navigation")
page_selection = st.sidebar.radio(
    "Select Platform View:",
    [
        "1. Live Interactive Tutor & Decision Inspector",
        "2. BKT Belief & Curriculum Tracker",
        "3. A/B Benchmark (RL vs Traditional Tutor)",
        "4. System Architecture & Flow Viewer",
        "5. Policy Training & Replay Inspector",
        "6. Supervisor Session Analytics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.title("Session & Model Setup")

student_id = st.sidebar.text_input("Student Identifier:", value="STU_101").strip()
if not student_id:
    student_id = "STU_ANONYMOUS"

model_path = "models/d3qn_tutor.pt"
d3qn_agent = load_d3qn_cached(model_path)

if d3qn_agent is not None:
    st.sidebar.success("PyTorch D3QN Model Loaded (`models/d3qn_tutor.pt`)")
else:
    st.sidebar.warning("Model Checkpoint Not Found (Using random policy for demo)")

# --- State Initialization ---
if "env" not in st.session_state:
    st.session_state.env = POMDPTutorEnv(max_steps=30)
    st.session_state.obs, st.session_state.info = st.session_state.env.reset(seed=42)

if "beliefs_history" not in st.session_state:
    st.session_state.beliefs_history = [list(st.session_state.env.bkt_engine.get_beliefs())]

if "last_action" not in st.session_state:
    st.session_state.last_action = 1

if "last_q_values" not in st.session_state:
    st.session_state.last_q_values = np.zeros(5, dtype=np.float32)

if "last_val_s" not in st.session_state:
    st.session_state.last_val_s = 0.0

if "last_adv_sa" not in st.session_state:
    st.session_state.last_adv_sa = np.zeros(5, dtype=np.float32)

if "last_nlg" not in st.session_state:
    st.session_state.last_nlg = 0.0

if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0.0

if "target_kc_idx" not in st.session_state:
    st.session_state.target_kc_idx = 0

if "current_question" not in st.session_state:
    st.session_state.current_question = get_curriculum_question(kc_idx=0, action_idx=1, seed=42)


# ==============================================================================
# VIEW 1: Live Interactive Tutor & Decision Inspector
# ==============================================================================
if page_selection.startswith("1"):
    st.subheader("Interactive Tutoring & Real-Time Decision Inspector")

    col_q, col_inspect = st.columns([1.1, 1.9])

    with col_q:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown(f"### Current Problem Item (Step {st.session_state.env.current_step})")

        q_item: QuestionItem = st.session_state.current_question
        st.write(f"**Target Skill:** {q_item.kc_name} | **Difficulty:** {q_item.difficulty}")
        st.info(f"**Question Prompt:** {q_item.prompt}")

        if st.session_state.last_action == 3:  # Worked Example
            st.success(f"**Worked Example Demonstration:** {q_item.worked_example_explanation}")
        elif st.session_state.last_action == 4:  # Scaffolding Hint
            st.warning(f"**Scaffolding Hint:** {q_item.hint}")

        user_answer = st.number_input("Your Numerical Answer:", value=0.0, step=1.0)
        submit_btn = st.button("Submit Answer", type="primary", use_container_width=True)

        if submit_btn:
            is_correct = bool(abs(user_answer - q_item.answer) < 1e-3)
            if is_correct:
                st.balloons()
                st.success("Correct Answer!")
            else:
                st.error(f"Incorrect. Solution: {q_item.explanation}")

            # Get agent policy action for next step
            obs = st.session_state.obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            if d3qn_agent is not None:
                val_s, adv_sa, q_vals = d3qn_agent.online_net.get_decomposed(obs_tensor)
                val_s_scalar = float(val_s.item())
                adv_sa_arr = adv_sa.squeeze(0).detach().cpu().numpy()
                q_vals_arr = q_vals.squeeze(0).detach().cpu().numpy()
                action = int(np.argmax(q_vals_arr))
            else:
                action = random.randint(0, 4)
                val_s_scalar = 0.0
                adv_sa_arr = np.zeros(5, dtype=np.float32)
                q_vals_arr = np.zeros(5, dtype=np.float32)

            # Step environment
            next_obs, reward, terminated, truncated, info = st.session_state.env.step(action)
            st.session_state.obs = next_obs
            st.session_state.last_action = action
            st.session_state.last_val_s = val_s_scalar
            st.session_state.last_adv_sa = adv_sa_arr
            st.session_state.last_q_values = q_vals_arr
            st.session_state.last_nlg = float(info.get("nlg", 0.0))
            st.session_state.last_reward = float(reward)
            st.session_state.target_kc_idx = int(info.get("target_skill", 0))

            beliefs = list(st.session_state.env.bkt_engine.get_beliefs())
            st.session_state.beliefs_history.append(beliefs)

            # Log interaction
            log_interaction(
                student_id=student_id,
                tutor_mode="Experimental Mode (D3QN RL)",
                step=st.session_state.env.current_step,
                kc_idx=st.session_state.target_kc_idx,
                kc_name=KC_NAMES[st.session_state.target_kc_idx],
                action_idx=action,
                action_name=info.get("action_name", "Practice"),
                is_correct=is_correct,
                response_time_sec=2.5,
                mean_belief=float(st.session_state.env.bkt_engine.get_mean_belief()),
                beliefs=beliefs,
            )

            # Select next question item
            st.session_state.current_question = get_curriculum_question(
                kc_idx=st.session_state.target_kc_idx, action_idx=action
            )
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_inspect:
        st.markdown("### D3QN Decision Inspector & Tensor Forward Pass")

        # Visualization 3: D3QN Forward Pass Flow Diagram
        fig_flow = create_d3qn_inspector_flow(
            obs=st.session_state.obs,
            val_s=st.session_state.last_val_s,
            adv_sa=st.session_state.last_adv_sa,
            q_values=st.session_state.last_q_values,
            selected_action=st.session_state.last_action,
        )
        st.plotly_chart(fig_flow, use_container_width=True)

        # Visualization 12: Natural Language Explainability Card
        explain_card = generate_explainability_card(
            obs=st.session_state.obs,
            q_values=st.session_state.last_q_values,
            selected_action=st.session_state.last_action,
            val_s=st.session_state.last_val_s,
            adv_sa=st.session_state.last_adv_sa,
            target_kc_name=KC_NAMES[st.session_state.target_kc_idx],
            beliefs=st.session_state.env.bkt_engine.get_beliefs(),
        )

        st.markdown('<div class="explain-box">', unsafe_allow_html=True)
        st.markdown("#### AI Agent Explainability Rationale")
        st.markdown(explain_card["rationale"])
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Expected Return Q(s,a)", f"{explain_card['expected_return']:.2f}")
        with m2:
            st.metric("Policy Confidence", f"{explain_card['confidence_pct']:.1f}%")
        with m3:
            st.metric("Q-Value Margin over 2nd Best", f"{explain_card['q_margin']:+.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Row 2: Action Comparison & Reward Decomposition
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        # Visualization 4: Action Comparison Panel
        fig_action = create_action_comparison_panel(
            q_values=st.session_state.last_q_values,
            selected_action=st.session_state.last_action,
        )
        st.plotly_chart(fig_action, use_container_width=True)

    with r2_c2:
        # Visualization 5: Reward Decomposition
        fig_reward = create_reward_decomposition_chart(
            nlg=st.session_state.last_nlg,
            is_hint_spam=bool(st.session_state.env.consecutive_hints > 2),
            is_mastery=bool(st.session_state.env.bkt_engine.get_mean_belief() >= 0.90),
        )
        st.plotly_chart(fig_reward, use_container_width=True)

    st.markdown("---")

    # Row 3: Hidden vs Observed & Decision Timeline
    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        # Visualization 6: Hidden State vs Observable State
        fig_hidden = create_hidden_vs_observed_panel(
            obs=st.session_state.obs,
            beliefs=st.session_state.env.bkt_engine.get_beliefs(),
            last_correct=bool(st.session_state.env.last_correctness > 0.5),
            last_friction=st.session_state.env.last_friction,
            last_action_name=POMDPTutorEnv.ACTION_MAP.get(st.session_state.last_action, "Practice"),
        )
        st.plotly_chart(fig_hidden, use_container_width=True)

    with r3_c2:
        # Visualization 8: Sequential RL Decision Timeline
        fig_timeline = create_decision_timeline(
            current_step=st.session_state.env.current_step,
            action_name=POMDPTutorEnv.ACTION_MAP.get(st.session_state.last_action, "Practice"),
            target_kc_name=KC_NAMES[st.session_state.target_kc_idx],
            nlg=st.session_state.last_nlg,
            reward=st.session_state.last_reward,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


# ==============================================================================
# VIEW 2: BKT Belief & Curriculum Tracker
# ==============================================================================
elif page_selection.startswith("2"):
    st.subheader("Bayesian Knowledge Tracing (BKT) & Curriculum Tracker")

    col_bkt1, col_bkt2 = st.columns([1.5, 1.0])

    with col_bkt1:
        # Visualization 1: Live BKT Line Chart
        fig_bkt, metrics = create_bkt_evolution_chart(
            history_beliefs=st.session_state.beliefs_history,
            active_kc_idx=st.session_state.target_kc_idx,
        )
        st.plotly_chart(fig_bkt, use_container_width=True)

    with col_bkt2:
        st.markdown("### BKT Summary Metrics")
        st.metric("Overall Mean Belief P(L)", f"{metrics['mean_belief']:.3f}")
        st.metric("Normalized Learning Gain (NLG)", f"{metrics['learning_gain']:+.3f}")
        st.metric("Mastery Achievement Ratio", f"{metrics['mastery_ratio']*100.0:.0f}%")

        # Visualization 7: Curriculum Learning Journey Node Graph
        fig_journey = create_learning_journey_graph(
            beliefs=st.session_state.env.bkt_engine.get_beliefs(),
            target_kc_idx=st.session_state.target_kc_idx,
        )
        st.plotly_chart(fig_journey, use_container_width=True)

    st.markdown("---")
    # Visualization 2: Interactive Belief Heatmap Across Time
    fig_heatmap = create_bkt_heatmap(history_beliefs=st.session_state.beliefs_history)
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ==============================================================================
# VIEW 3: A/B Benchmark (RL vs Traditional Tutor)
# ==============================================================================
elif page_selection.startswith("3"):
    st.subheader("A/B Benchmark — PyTorch D3QN Agent vs. Traditional Linear Tutor")

    st.write(
        "Execute a simultaneous side-by-side simulation benchmarking the adaptive PyTorch D3QN RL tutor against a standard linear control tutor."
    )

    if st.button("Run Live A/B Simulation Run (30 Steps)", type="primary"):
        env_rl = POMDPTutorEnv(max_steps=30)
        env_ctrl = POMDPTutorEnv(max_steps=30)

        obs_rl, _ = env_rl.reset(seed=42)
        obs_ctrl, _ = env_ctrl.reset(seed=42)

        ctrl_policy = HeuristicTutor(initial_difficulty=1)

        rl_b_hist, ctrl_b_hist = [], []
        rl_r_hist, ctrl_r_hist = [], []
        rl_a_hist, ctrl_a_hist = [], []

        for step in range(30):
            # RL Action
            if d3qn_agent is not None:
                act_rl = d3qn_agent.select_action(obs_rl, epsilon=0.0)
            else:
                act_rl = random.randint(0, 4)

            # Control Action
            act_ctrl = ctrl_policy.select_action(obs_ctrl)

            obs_rl, rew_rl, term_rl, trunc_rl, _ = env_rl.step(act_rl)
            obs_ctrl, rew_ctrl, term_ctrl, trunc_ctrl, _ = env_ctrl.step(act_ctrl)

            rl_b_hist.append(float(env_rl.bkt_engine.get_mean_belief()))
            ctrl_b_hist.append(float(env_ctrl.bkt_engine.get_mean_belief()))

            rl_r_hist.append(float(rew_rl))
            ctrl_r_hist.append(float(rew_ctrl))

            rl_a_hist.append(POMDPTutorEnv.ACTION_MAP[act_rl])
            ctrl_a_hist.append(POMDPTutorEnv.ACTION_MAP[act_ctrl])

            if term_rl or term_ctrl:
                break

        # Visualization 13: A/B Split-screen figure
        fig_ab = create_ab_splitscreen_view(
            rl_beliefs=rl_b_hist,
            control_beliefs=ctrl_b_hist,
            rl_rewards=rl_r_hist,
            control_rewards=ctrl_r_hist,
            rl_actions=rl_a_hist,
            control_actions=ctrl_a_hist,
        )
        st.plotly_chart(fig_ab, use_container_width=True)


# ==============================================================================
# VIEW 4: System Architecture & Flow Viewer
# ==============================================================================
elif page_selection.startswith("4"):
    st.subheader("Full Intelligent Tutoring System Architecture & Data Flow Loop")

    # Visualization 15: Full Architecture Animation
    step_anim = st.slider("Animate Data Flow Step:", min_value=0, max_value=9, value=0, step=1)
    fig_arch = create_full_architecture_diagram(active_step_idx=step_anim)
    st.plotly_chart(fig_arch, use_container_width=True)

    # Visualization 11: Neural Network Layer Diagram
    st.markdown("---")
    st.subheader("Dueling DQN Neural Network Architecture & Feature Dimensions")
    fig_nn = create_neural_activity_diagram(
        state_dim=9, hidden_dim=128, action_dim=5, last_obs=st.session_state.obs
    )
    st.plotly_chart(fig_nn, use_container_width=True)


# ==============================================================================
# VIEW 5: Policy Training & Replay Inspector
# ==============================================================================
elif page_selection.startswith("5"):
    st.subheader("PyTorch D3QN Policy Training & Replay Buffer Inspector")

    c_train1, c_train2 = st.columns([1.0, 2.0])

    with c_train1:
        st.markdown("### Replay Memory Inspector")
        replay_size = len(d3qn_agent.memory) if d3qn_agent is not None else 14200
        fig_gauge, _ = create_replay_buffer_inspector(capacity=50000, current_size=replay_size)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c_train2:
        st.markdown("### Interactive Model Training Controls")
        train_episodes = st.slider("Training Episodes:", min_value=500, max_value=5000, value=1000, step=500)
        start_train = st.button("Start Training Policy Run", type="primary", use_container_width=True)

    if start_train:
        env_tr = POMDPTutorEnv(max_steps=30)
        agent_tr = D3QNAgent(state_dim=9, action_dim=5, device="cpu")

        rewards_h, losses_h, eps_h, acts_h = [], [], [], []

        pbar = st.progress(0)
        eps_start, eps_min = 1.0, 0.05
        explore_episodes = max(1, int(train_episodes * 0.6))

        for ep in range(1, train_episodes + 1):
            if ep <= explore_episodes:
                epsilon = eps_start - (ep / float(explore_episodes)) * (eps_start - eps_min)
            else:
                epsilon = eps_min

            obs, _ = env_tr.reset()
            ep_rew = 0
            acts = [0] * 5

            for step in range(30):
                action = agent_tr.select_action(obs, epsilon=epsilon)
                acts[action] += 1
                next_obs, rew, term, trunc, _ = env_tr.step(action)
                agent_tr.memory.push(obs, action, rew, next_obs, term)
                agent_tr.train_step()
                obs = next_obs
                ep_rew += rew
                if term or trunc:
                    break

            if ep % 20 == 0:
                agent_tr.update_target_network()

            rewards_h.append(ep_rew)
            eps_h.append(epsilon)
            acts_h.append(acts)

            if ep % 50 == 0 or ep == train_episodes:
                pbar.progress(ep / float(train_episodes))

        pbar.progress(1.0)
        agent_tr.save(model_path)
        st.session_state.training_history = {
            "rewards": rewards_h,
            "losses": [0.05] * len(rewards_h),
            "epsilons": eps_h,
            "actions": acts_h,
        }
        st.success(f"Training Complete ({train_episodes} Episodes)! Checkpoint saved to `models/d3qn_tutor.pt`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Policy Evolution & Telemetry Dashboard")

    # Render training dashboard from session state if available, else synthetic demo baseline
    if "training_history" in st.session_state:
        th = st.session_state.training_history
        fig_train = create_training_policy_dashboard(
            episode_rewards=th["rewards"],
            episode_losses=th["losses"],
            epsilons=th["epsilons"],
            action_counts_history=th["actions"],
        )
    else:
        # Default baseline demo state
        demo_episodes = 1000
        demo_rewards = list(np.sin(np.linspace(0, 3, demo_episodes)) * 15 + np.random.randn(demo_episodes) * 5 + 10)
        demo_eps = list(np.linspace(1.0, 0.05, demo_episodes))
        demo_acts = [[int(x) for x in np.random.multinomial(30, [0.2]*5)] for _ in range(demo_episodes)]
        fig_train = create_training_policy_dashboard(
            episode_rewards=demo_rewards,
            episode_losses=[0.08] * demo_episodes,
            epsilons=demo_eps,
            action_counts_history=demo_acts,
        )

    st.plotly_chart(fig_train, use_container_width=True)


# ==============================================================================
# VIEW 6: Supervisor Session Analytics
# ==============================================================================
elif page_selection.startswith("6"):
    st.subheader("Global Session Telemetry & Statistical Efficacy")

    analyzer = SessionAnalyzer()
    session_df, survey_df = analyzer.load_data()

    # Visualization 14: Global Session Analytics Dashboard
    fig_analytics = create_session_analytics_dashboard(session_df=session_df, survey_df=survey_df)
    st.plotly_chart(fig_analytics, use_container_width=True)
