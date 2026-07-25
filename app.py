"""Advanced Streamlit Web Application for POMDP-BKT Intelligent Tutoring System.

This application provides:
1. POMDP-BKT Simulation Dashboard: Benchmarks PyTorch D3QN agent vs baselines with NLG metrics.
2. Live Interactive Tutor & A/B Testing: Real-time BKT 4-skill belief tracking, A/B Testing modes
   (Control Mode vs Experimental Mode), CSV interaction logging, and PyTorch D3QN explainability.
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


# Force PyTorch single-threading to prevent Streamlit watcher thread segmentation faults on macOS
torch.set_num_threads(1)



from src.bkt_engine import BKTEngine
from src.pomdp_env import POMDPTutorEnv
from src.baselines import RandomTutor
from src.d3qn_agent import D3QNAgent, D3QNTutor, train_d3qn
from src.evaluator import POMDPHeuristicTutor, evaluate_pomdp_tutor, run_pomdp_benchmark
from src.curriculum import KC_NAMES, QuestionItem, get_curriculum_question
from src.logger import log_interaction, log_affective_survey, ensure_log_files_exist
from src.analytics import SessionAnalyzer



# --- Model Cache ---
@st.cache_resource
def load_d3qn_cached(model_path: str = "models/d3qn_tutor.pt"):
    """Loads and caches the PyTorch D3QN agent.

    Args:
        model_path: Path to model pt file.

    Returns:
        D3QNAgent instance or None.
    """
    if os.path.exists(model_path):
        agent = D3QNAgent(state_dim=9, action_dim=5, device="cpu")
        agent.load(model_path)
        agent.online_net.eval()
        return agent
    return None


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="POMDP Intelligent Tutoring System",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    /* Global Base Dark Theme */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #09090B !important;
        color: #FAFAFA !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #121215 !important;
        border-right: 1px solid #27272A !important;
    }

    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseGlow {
        0% { border-color: rgba(255, 255, 255, 0.15); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.05); }
        50% { border-color: rgba(255, 255, 255, 0.45); box-shadow: 0 0 15px 0 rgba(255, 255, 255, 0.1); }
        100% { border-color: rgba(255, 255, 255, 0.15); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.05); }
    }

    /* Headers */
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(180deg, #FFFFFF 0%, #A1A1AA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        animation: fadeInUp 0.5s ease-out;
    }

    .sub-header {
        font-size: 1.02rem;
        color: #A1A1AA;
        margin-bottom: 1.8rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }

    /* Modern Monochrome Containers */
    .question-box {
        background: #18181B;
        border: 1px solid #27272A;
        border-left: 4px solid #FFFFFF;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .question-box:hover {
        transform: translateY(-2px);
        border-color: #3F3F46;
    }

    .worked-example-box {
        background: #141417;
        border: 1px solid #3F3F46;
        border-left: 4px solid #E4E4E7;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        color: #F4F4F5;
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .hint-box {
        background-color: #121215;
        border: 1px solid #27272A;
        border-left: 4px solid #A1A1AA;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        color: #D4D4D8;
        animation: fadeInUp 0.3s ease-out;
    }

    /* High Contrast Mode Badges */
    .mode-badge-control {
        background-color: #27272A;
        color: #E4E4E7;
        border: 1px solid #3F3F46;
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }

    .mode-badge-exp {
        background-color: #FFFFFF;
        color: #09090B;
        border: 1px solid #FFFFFF;
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.02em;
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.2);
    }

    /* Metric cards styling */
    [data-testid="stMetric"] {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #3F3F46;
    }
    [data-testid="stMetricLabel"] {
        color: #A1A1AA !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Streamlit Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #FFFFFF !important;
    }

    /* Streamlit Buttons */
    .stButton > button {
        background-color: #18181B !important;
        color: #FFFFFF !important;
        border: 1px solid #3F3F46 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #09090B !important;
        border-color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.15) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #FFFFFF !important;
        color: #09090B !important;
        border: 1px solid #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #E4E4E7 !important;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize CSV log directory
ensure_log_files_exist()

# --- App Header ---
st.markdown('<div class="main-header">POMDP Intelligent Tutoring System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Bayesian Knowledge Tracing (BKT) and PyTorch Dueling Double Deep Q-Network (D3QN) Experimental Evaluation Platform</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Navigation Menu ---
st.sidebar.title("Navigation Menu")
page_selection = st.sidebar.radio(
    "Select Page:",
    ["Tutoring Session", "Supervisor Analytics"],
)

st.sidebar.markdown("---")
st.sidebar.title("Session & Control Configuration")

student_id = st.sidebar.text_input("Student Identifier:", value="STU_101").strip()
if not student_id:
    student_id = "STU_ANONYMOUS"

tutor_mode = st.sidebar.selectbox(
    "Select Tutor Mode:",
    [
        "Experimental Mode (D3QN RL Tutor)",
        "Control Mode (Standard Linear Tutor)",
    ],
)

app_mode = "Live Interactive Tutor & Explainability"
if page_selection == "Tutoring Session":
    app_mode = st.sidebar.radio(
        "Select View Mode:",
        ["Live Interactive Tutor & Explainability", "POMDP-BKT Simulation Dashboard"],
    )

model_path = "models/d3qn_tutor.pt"
d3qn_agent = load_d3qn_cached(model_path)

st.sidebar.markdown("---")
st.sidebar.subheader("PyTorch D3QN Model & Training Controls")
if d3qn_agent is not None:
    st.sidebar.success("PyTorch D3QN Model Loaded (`models/d3qn_tutor.pt`)")
else:
    st.sidebar.warning("Model Checkpoint Not Found")

with st.sidebar.expander("Advanced Model Training Controls", expanded=d3qn_agent is None):
    st.write("Configure hyperparameters to execute PyTorch D3QN policy training:")
    train_episodes = st.slider("Total Episodes:", min_value=1000, max_value=25000, value=5000, step=1000)
    train_lr = st.select_slider(
        "Learning Rate:",
        options=[0.00005, 0.0001, 0.0003, 0.0005, 0.001],
        value=0.0001,
        format_func=lambda x: f"{x:.5f}",
    )
    train_buffer = st.selectbox("Replay Buffer Capacity:", options=[50000, 100000, 200000], index=1)
    train_seed = st.number_input("Training Seed:", min_value=1, max_value=9999, value=42, step=1)

    if st.button("Execute PyTorch D3QN Training Run", type="primary", use_container_width=True):
        st.write("---")
        st.write("#### Training Progress Telemetry")
        pbar = st.progress(0)
        status_text = st.empty()
        metric_c1, metric_c2 = st.columns(2)
        with metric_c1:
            m_rew = st.empty()
        with metric_c2:
            m_eps = st.empty()

        def ui_callback(ep: int, total: int, reward: float, eps: float, buf_size: int) -> None:
            pct = int((ep / float(total)) * 100)
            pbar.progress(pct)
            status_text.text(f"Episode {ep}/{total} ({pct}%) | Buffer: {buf_size}")
            m_rew.metric("Last Episode Reward", f"{reward:.2f}")
            m_eps.metric("Current Epsilon", f"{eps:.3f}")

        with st.spinner(f"Training PyTorch D3QN Agent ({train_episodes} episodes)..."):
            env = POMDPTutorEnv(max_steps=30)
            train_d3qn(
                env=env,
                total_episodes=train_episodes,
                learning_rate=train_lr,
                buffer_capacity=train_buffer,
                save_path=model_path,
                seed=train_seed,
                callback=ui_callback,
            )
            st.cache_resource.clear()
            st.success(f"Training complete! Model saved to `{model_path}`.")
            time.sleep(1)
            st.rerun()


st.sidebar.markdown("---")
st.sidebar.info(
    f"**Active Session Context:**\n"
    f"- **Student:** `{student_id}`\n"
    f"- **Tutor Mode:** {'D3QN Adaptive' if 'Experimental' in tutor_mode else 'Linear Fixed'}\n"
    f"- **Log File:** `data/session_logs.csv`"
)


# ==============================================================================
# SUPERVISOR ANALYTICS PAGE
# ==============================================================================
if page_selection == "Supervisor Analytics":
    st.header("Supervisor Telemetry Analytics Dashboard")
    st.write(
        "Statistical telemetry analysis comparing the **RL Experimental Group (D3QN Agent)** against the **Control Group (Standard Linear Tutor)**."
    )

    analyzer = SessionAnalyzer()
    gains_df = analyzer.get_learning_gains()
    steps_df = analyzer.get_steps_to_mastery()
    affective_df = analyzer.get_affective_averages()
    actions_df = analyzer.get_action_distribution()

    # Extract KPI Values
    rl_nlg = 0.0
    control_nlg = 0.0
    if not gains_df.empty:
        rl_match = gains_df[gains_df["Mode"] == "RL Mode"]
        ctrl_match = gains_df[gains_df["Mode"] == "Control Mode"]
        if not rl_match.empty:
            rl_nlg = float(rl_match.iloc[0]["Mean_NLG"])
        if not ctrl_match.empty:
            control_nlg = float(ctrl_match.iloc[0]["Mean_NLG"])

    rl_steps = 0.0
    if not steps_df.empty:
        rl_steps_df = steps_df[steps_df["Mode"] == "RL Mode"]
        if not rl_steps_df.empty:
            rl_steps = float(rl_steps_df["steps"].mean())

    total_sessions = len(steps_df) if not steps_df.empty else 0
    total_surveys = int(affective_df["Survey_Count"].sum()) if not affective_df.empty else 0

    # Top KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Sessions Logged", f"{total_sessions}")
    with kpi2:
        st.metric("RL Mean NLG", f"{rl_nlg:.3f}")
    with kpi3:
        st.metric("Control Mean NLG", f"{control_nlg:.3f}")
    with kpi4:
        st.metric("RL Avg Steps to Mastery", f"{rl_steps:.1f}")
    with kpi5:
        st.metric("Affective Surveys Logged", f"{total_surveys}")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Interactive Plotly Analytics Section
    # --------------------------------------------------------------------------
    st.subheader("Comparative Statistical Efficacy and Telemetry")
    chart_col1, chart_col2 = st.columns(2)

    # Chart 1: Learning Efficacy (Mean NLG Comparison)
    with chart_col1:
        st.markdown("#### 1. Learning Efficacy (Mean NLG)")
        if not gains_df.empty:
            fig_nlg = px.bar(
                gains_df,
                x="Mode",
                y="Mean_NLG",
                error_y="Std_NLG",
                color="Mode",
                color_discrete_map={"RL Mode": "#FFFFFF", "Control Mode": "#71717A"},
                title="Mean Normalized Learning Gain (NLG) by Mode",
                labels={"Mean_NLG": "Mean NLG", "Mode": "Tutor Group"},
                template="plotly_dark",
            )
            fig_nlg.update_layout(
                paper_bgcolor="#18181B",
                plot_bgcolor="#18181B",
                font=dict(color="#FAFAFA"),
                showlegend=False,
            )
            st.plotly_chart(fig_nlg, use_container_width=True)
        else:
            st.info("No learning gain telemetry available yet.")

    # Chart 2: Learning Efficiency (Steps to Mastery Distribution Box Plot)
    with chart_col2:
        st.markdown("#### 2. Learning Efficiency (Time-on-Task)")
        if not steps_df.empty:
            fig_steps = px.box(
                steps_df,
                x="Mode",
                y="steps",
                color="Mode",
                color_discrete_map={"RL Mode": "#FFFFFF", "Control Mode": "#71717A"},
                title="Distribution of Steps to Mastery",
                labels={"steps": "Steps to Mastery", "Mode": "Tutor Group"},
                template="plotly_dark",
            )
            fig_steps.update_layout(
                paper_bgcolor="#18181B",
                plot_bgcolor="#18181B",
                font=dict(color="#FAFAFA"),
                showlegend=False,
            )
            st.plotly_chart(fig_steps, use_container_width=True)
        else:
            st.info("No step telemetry available yet.")

    st.markdown("---")

    chart_col3, chart_col4 = st.columns(2)

    # Chart 3: Agent Explainability (Action Distribution Stacked Bar Chart)
    with chart_col3:
        st.markdown("#### 3. Agent Explainability (Action Distribution)")
        if not actions_df.empty:
            fig_act = px.bar(
                actions_df,
                x="action_name",
                y="percentage",
                color="Mode",
                barmode="group",
                color_discrete_map={"RL Mode": "#FFFFFF", "Control Mode": "#71717A"},
                title="Action Choice Percentage Breakdown",
                labels={"percentage": "Percentage (%)", "action_name": "Pedagogical Action", "Mode": "Group"},
                template="plotly_dark",
            )
            fig_act.update_layout(
                paper_bgcolor="#18181B",
                plot_bgcolor="#18181B",
                font=dict(color="#FAFAFA"),
            )
            st.plotly_chart(fig_act, use_container_width=True)
        else:
            st.info("No action selection telemetry available yet.")

    # Chart 4: Affective Impact (Engagement & Frustration Comparison)
    with chart_col4:
        st.markdown("#### 4. Affective Impact (Survey Telemetry)")
        if not affective_df.empty:
            aff_melted = affective_df.melt(
                id_vars=["Mode"],
                value_vars=["Mean_Engagement", "Mean_Frustration"],
                var_name="Affective_Variable",
                value_name="Score",
            )
            aff_melted["Affective_Variable"] = aff_melted["Affective_Variable"].replace({
                "Mean_Engagement": "Engagement (1-5)",
                "Mean_Frustration": "Frustration (1-5)",
            })

            fig_aff = px.bar(
                aff_melted,
                x="Affective_Variable",
                y="Score",
                color="Mode",
                barmode="group",
                color_discrete_map={"RL Mode": "#FFFFFF", "Control Mode": "#71717A"},
                title="Self-Reported Affective Metrics Comparison",
                labels={"Score": "Mean Likert Score (1-5)", "Affective_Variable": "Variable"},
                template="plotly_dark",
            )
            fig_aff.update_layout(
                paper_bgcolor="#18181B",
                plot_bgcolor="#18181B",
                font=dict(color="#FAFAFA"),
                yaxis=dict(range=[0, 5.5]),
            )
            st.plotly_chart(fig_aff, use_container_width=True)
        else:
            st.info("No affective survey telemetry available yet.")

    st.markdown("---")

    # Raw telemetry data inspection table
    with st.expander("View Raw Telemetry Log Data", expanded=False):
        sess_df, sur_df = analyzer.load_data()
        st.markdown("##### Session Interactions (`data/session_logs.csv`)")
        st.dataframe(sess_df, use_container_width=True)

        st.markdown("##### Affective Surveys (`data/affective_surveys.csv`)")
        st.dataframe(sur_df, use_container_width=True)

    st.stop()



# ==============================================================================
# MODE 1: POMDP-BKT SIMULATION DASHBOARD
# ==============================================================================
elif app_mode == "POMDP-BKT Simulation Dashboard":

    st.header("Comparative Policy Evaluation Dashboard")
    st.write(
        "Evaluate the **PyTorch Dueling Double DQN (D3QN)** policy against baseline tutors on the 4-KC POMDP environment."
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        eval_episodes = st.slider("Evaluation Episodes per Policy:", min_value=10, max_value=100, value=30, step=10)
    with c2:
        eval_seed = st.number_input("Random Seed:", min_value=1, max_value=9999, value=42, step=1)
    with c3:
        st.write("")
        st.write("")
        run_btn = st.button("Run Benchmark Evaluation", type="primary", use_container_width=True)

    if run_btn or "pomdp_results" in st.session_state:
        if run_btn or "pomdp_results" not in st.session_state:
            with st.spinner("Running POMDP-BKT evaluation benchmark..."):
                if d3qn_agent is None:
                    env = POMDPTutorEnv(max_steps=30)
                    train_d3qn(env, total_episodes=5000, save_path=model_path, seed=eval_seed)
                    st.cache_resource.clear()
                    d3qn_agent = load_d3qn_cached(model_path)

                df, raw = run_pomdp_benchmark(n_episodes=eval_episodes, seed=eval_seed, model_path=model_path)
                st.session_state["pomdp_results"] = (df, raw)

        df, results = st.session_state["pomdp_results"]

        # Metric summary cards
        st.subheader("Policy Performance Summary")
        col_m1, col_m2, col_m3 = st.columns(3)

        for col, (policy_name, res) in zip([col_m1, col_m2, col_m3], results.items()):
            with col:
                st.markdown(f"### {policy_name}")
                st.metric("Mean Reward", f"{res['mean_reward']:.2f}", delta=f"± {res['std_reward']:.2f}")
                st.metric("Mean Normalized Learning Gain (NLG)", f"{res['mean_nlg']:.3f}", delta=f"± {res['std_nlg']:.3f}")
                st.metric("Final Skill Belief P(L_t)", f"{res['mean_final_belief']:.3f}", delta=f"± {res['std_final_belief']:.3f}")
                st.metric("Mastery Rate", f"{res['mastery_rate'] * 100:.1f}%")

        st.markdown("---")

        # Line chart of belief trajectory
        st.subheader("Multi-Skill Belief Trajectory Progression")
        max_len = 31
        traj_df = pd.DataFrame({"Step": list(range(max_len))})

        for policy_name, res in results.items():
            padded = [
                t + [t[-1]] * (max_len - len(t)) if len(t) < max_len else t[:max_len]
                for t in res["belief_trajectories"]
            ]
            traj_df[policy_name] = np.mean(padded, axis=0)

        st.line_chart(traj_df.set_index("Step"), use_container_width=True)

        # Artifact plots display
        st.subheader("Generated Analysis Artifacts")
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            if os.path.exists("artifacts/pomdp_belief_trajectories.png"):
                st.image("artifacts/pomdp_belief_trajectories.png", caption="Belief Trajectories Comparison")
        with img_c2:
            if os.path.exists("artifacts/pomdp_nlg_comparison.png"):
                st.image("artifacts/pomdp_nlg_comparison.png", caption="Reward vs. NLG Comparison")


# ==============================================================================
# MODE 2: LIVE INTERACTIVE TUTOR & EXPLAINABILITY (A/B TESTING)
# ==============================================================================
elif app_mode == "Live Interactive Tutor & Explainability":
    st.header("Interactive Live Math Tutor")

    is_experimental = "Experimental" in tutor_mode
    mode_badge_html = (
        '<div class="mode-badge-exp">Experimental Mode (D3QN RL Tutor)</div>'
        if is_experimental
        else '<div class="mode-badge-control">Control Mode (Standard Linear Tutor)</div>'
    )
    st.markdown(f"Student ID: **`{student_id}`** | Tutor Mode: {mode_badge_html}", unsafe_allow_html=True)
    st.write("")

    if is_experimental and d3qn_agent is None:
        st.error("Trained D3QN Model (`models/d3qn_tutor.pt`) is missing. Please train the model from the sidebar first.")
        st.stop()

    d3qn_tutor = D3QNTutor(d3qn_agent) if d3qn_agent is not None else None

    # Initialize Session State
    if (
        "pomdp_state" not in st.session_state
        or st.session_state["pomdp_state"].get("student_id") != student_id
        or st.session_state["pomdp_state"].get("tutor_mode") != tutor_mode
    ):
        bkt = BKTEngine(seed=42)
        init_beliefs = bkt.reset()

        # Select initial target skill and action
        target_kc = 0
        initial_action = 0

        if is_experimental and d3qn_tutor is not None:
            # 9D initial observation
            obs = np.array([init_beliefs[0], init_beliefs[1], init_beliefs[2], init_beliefs[3], float(np.mean(init_beliefs)), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            initial_action = d3qn_tutor.select_action(obs)
            target_kc = int(np.argmin(init_beliefs))

        first_q = get_curriculum_question(target_kc, initial_action, seed=random.randint(1, 1000))

        st.session_state["pomdp_state"] = {
            "student_id": student_id,
            "tutor_mode": tutor_mode,
            "bkt_engine": bkt,
            "recent_history": [],
            "last_action": initial_action,
            "last_correctness": 0.0,
            "last_friction": 0.0,
            "current_step": 0,
            "max_steps": 20,
            "score": 0,
            "current_action": initial_action,
            "target_skill_idx": target_kc,
            "current_item": first_q,
            "step_start_time": time.time(),
            "last_feedback": None,
        }

    pstate = st.session_state["pomdp_state"]
    bkt = pstate["bkt_engine"]
    beliefs = bkt.get_beliefs()
    mean_b = bkt.get_mean_belief()
    curr_item: QuestionItem = pstate["current_item"]

    # Header Controls
    col_hdr1, col_hdr2, col_hdr3 = st.columns([4, 1.5, 2.0])
    with col_hdr2:
        if st.button("Restart Session"):
            del st.session_state["pomdp_state"]
            st.rerun()
    with col_hdr3:
        if st.button("End Session and Complete Survey", type="secondary"):
            pstate["session_ended"] = True
            st.rerun()

    st.markdown("---")

    # --- Live Multi-Skill BKT Belief Dashboard (4 KCs) ---
    st.subheader("Bayesian Knowledge Tracing (BKT) 4-KC Belief State Tracking")
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)

    skill_names = bkt.skills
    with b_col1:
        st.metric(skill_names[0], f"{beliefs[0]:.2f}")
        st.progress(float(beliefs[0]))
    with b_col2:
        st.metric(skill_names[1], f"{beliefs[1]:.2f}")
        st.progress(float(beliefs[1]))
    with b_col3:
        st.metric(skill_names[2], f"{beliefs[2]:.2f}")
        st.progress(float(beliefs[2]))
    with b_col4:
        st.metric(skill_names[3], f"{beliefs[3]:.2f}")
        st.progress(float(beliefs[3]))
    with b_col5:
        st.metric("Overall Mean Belief P(L_t)", f"{mean_b:.2f}")
        st.progress(float(mean_b))

    st.markdown("---")

    # Construct Current 9D State Vector for Explainability
    rolling_acc = float(np.mean(pstate["recent_history"])) if pstate["recent_history"] else 0.0
    norm_step = min(pstate["current_step"] / float(pstate["max_steps"]), 1.0)
    norm_last_act = pstate["last_action"] / 4.0

    current_obs = np.array(
        [beliefs[0], beliefs[1], beliefs[2], beliefs[3], mean_b, rolling_acc, pstate["last_friction"], norm_step, norm_last_act],
        dtype=np.float32,
    )

    # --- D3QN AI Explainability Panel (Displayed in Experimental Mode) ---
    if is_experimental and d3qn_tutor is not None:
        v_s, adv_list, q_list = d3qn_tutor.predict_explainability(current_obs)
        with st.expander("**D3QN Neural Network Explainability & Action Q-Values**", expanded=True):
            ex_col1, ex_col2 = st.columns([1, 2])
            with ex_col1:
                st.markdown("#### State Value V(s)")
                st.metric("V(s) Expected Return", f"{v_s:.3f}")
                st.caption("Estimated cumulative learning return from current 9D student state.")

            with ex_col2:
                st.markdown("#### Action Q-Values Q(s, a) Breakdown")
                action_labels = ["Easy Problem", "Medium Problem", "Hard Problem", "Worked Example", "Scaffolding Hint"]
                q_df = pd.DataFrame({
                    "Action": action_labels,
                    "Advantage A(s,a)": [round(a, 3) for a in adv_list],
                    "Q-Value Q(s,a)": [round(q, 3) for q in q_list],
                }).set_index("Action")
                st.bar_chart(q_df["Q-Value Q(s,a)"], use_container_width=True)

    action_names = ["Easy Problem", "Medium Problem", "Hard Problem", "Worked Example", "Scaffolding Hint"]
    curr_act_idx = pstate["current_action"]
    curr_act_name = action_names[curr_act_idx]

    # Mastery Goal Banner
    if mean_b < 0.90 and not pstate.get("session_ended", False):
        st.info(f"**Mastery Threshold Target:** Reaching Mean Skill Belief $\\bar{{b}}_t \\ge 0.90$ | Current: **{mean_b:.2f}** ({min(100.0, (mean_b/0.90)*100):.1f}% of target)")

    # Check Session Termination or End Session Trigger
    if mean_b >= 0.90 or pstate["current_step"] >= pstate["max_steps"] or pstate.get("session_ended", False):
        st.success(
            f"**SESSION COMPLETED**\n\n"
            f"- **Student ID:** `{student_id}` | **Mode:** `{tutor_mode}`\n"
            f"- **Final Mean Skill Belief:** **{mean_b:.3f}** (Threshold ≥ 0.900)\n"
            f"- **Steps Attempted:** **{pstate['current_step']} steps**\n"
            f"- **Mastery Status:** {'Mastery Threshold Exceeded' if mean_b >= 0.90 else 'Max Steps Reached'}"
        )

        st.markdown("---")
        st.subheader("Post-Session Affective Evaluation Survey")
        st.write("Please complete this evaluation survey to record affective interaction telemetry.")

        if pstate.get("survey_submitted", False):
            st.success("Your affective survey response and session data have been logged to `data/affective_surveys.csv`.")
            if st.button("Start New Session", type="primary"):
                del st.session_state["pomdp_state"]
                st.rerun()
        else:
            with st.form(key="affective_survey_form"):
                engagement = st.select_slider(
                    "1. Rate the level of engagement during this session (1-5):",
                    options=[1, 2, 3, 4, 5],
                    value=4,
                    format_func=lambda x: f"{x} - " + {1: "Low Engagement", 2: "Slightly Engaging", 3: "Neutral", 4: "Engaging", 5: "High Engagement"}[x],
                )

                frustration = st.select_slider(
                    "2. Rate the level of frustration experienced during difficult tasks (1-5):",
                    options=[1, 2, 3, 4, 5],
                    value=2,
                    format_func=lambda x: f"{x} - " + {1: "Low Frustration", 2: "Slightly Frustrated", 3: "Moderate Frustration", 4: "Frustrated", 5: "High Frustration"}[x],
                )

                pacing = st.radio(
                    "3. Select pacing feedback:",
                    options=["Too Slow", "Optimal Pacing", "Too Fast"],
                    index=1,
                    horizontal=True,
                )

                comments = st.text_area("4. Additional observations or technical feedback (optional):", value="")

                submit_survey_btn = st.form_submit_button("Submit Survey and Log Data", type="primary")

            if submit_survey_btn:
                log_affective_survey(
                    student_id=student_id,
                    tutor_mode=tutor_mode,
                    engagement_score=engagement,
                    frustration_score=frustration,
                    pacing_feedback=pacing,
                    comments=comments,
                )
                pstate["survey_submitted"] = True
                st.rerun()

        st.stop()


    # Feedback Banner
    if pstate["last_feedback"] is not None:
        fb = pstate["last_feedback"]
        if fb["is_correct"]:
            st.success(f"**Response Correct.** Solution: *{fb['explanation']}* | Skill Belief Gain: **+{fb['delta_b']:.3f}**")
        else:
            st.error(f"**Response Incorrect.** Expected Answer: **{fb['correct_ans']}**. Solution: *{fb['explanation']}*")

    # Present Action Content (Worked Example vs Problem vs Hint)
    st.markdown(
        f"### Step {pstate['current_step'] + 1} | Target KC: **{curr_item.kc_name}** | Action: **{curr_act_name}**"
    )


    if curr_act_idx == 3:  # Worked Example Action
        st.markdown(
            f'<div class="worked-example-box">'
            f'<h3>{curr_item.worked_example_prompt}</h3>'
            f'<p>{curr_item.worked_example_explanation}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Confirm Worked Example Completion", type="primary"):

            resp_time = time.time() - pstate["step_start_time"]
            prev_b = bkt.get_mean_belief()
            t_kc = pstate["target_skill_idx"]
            bkt.update_belief(t_kc, is_correct=True, receives_scaffolding=True)
            new_b = bkt.get_mean_belief()

            # Log interaction to CSV
            log_interaction(
                student_id=student_id,
                tutor_mode=tutor_mode,
                step=pstate["current_step"] + 1,
                kc_idx=t_kc,
                kc_name=KC_NAMES[t_kc],
                action_idx=3,
                action_name=curr_act_name,
                is_correct=True,
                response_time_sec=resp_time,
                mean_belief=new_b,
                beliefs=bkt.get_beliefs().tolist(),
            )

            pstate["current_step"] += 1
            pstate["last_action"] = 3
            pstate["last_friction"] = 0.2
            pstate["last_feedback"] = {
                "is_correct": True,
                "correct_ans": "Studied Example",
                "explanation": curr_item.worked_example_explanation,
                "delta_b": new_b - prev_b,
            }

            # Select Next Action & KC
            if is_experimental and d3qn_tutor is not None:
                next_obs = np.array(
                    [bkt.beliefs[0], bkt.beliefs[1], bkt.beliefs[2], bkt.beliefs[3], new_b, 1.0, 0.2, min(pstate["current_step"]/30.0, 1.0), 3.0/4.0],
                    dtype=np.float32,
                )
                next_act = d3qn_tutor.select_action(next_obs)
                next_kc = int(np.argmin(bkt.get_beliefs()))
            else:  # Control Mode: Linear sequence
                next_kc = min(3, pstate["current_step"] // 5)
                next_act = (pstate["current_step"] % 3)

            pstate["current_action"] = next_act
            pstate["target_skill_idx"] = next_kc
            pstate["current_item"] = get_curriculum_question(next_kc, next_act, seed=random.randint(1, 10000))
            pstate["step_start_time"] = time.time()
            st.rerun()

    else:  # Practice Problem (0, 1, 2) or Scaffolding Hint (4)
        if curr_act_idx == 4 and curr_item.hint:
            st.markdown(
                f'<div class="hint-box"><h4>{curr_item.hint}</h4></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="question-box"><h2>{curr_item.prompt}</h2></div>',
            unsafe_allow_html=True,
        )

        with st.form(key="curriculum_ans_form", clear_on_submit=True):
            user_input_str = st.text_input("Enter your numerical answer:", key="curr_ans_input")
            submit_btn = st.form_submit_button("Submit Answer", type="primary")

        if submit_btn:
            cleaned = user_input_str.strip()
            if not cleaned:
                st.warning("Please enter a numerical answer before submitting.")
            else:
                try:
                    user_val = float(cleaned)
                    expected_val = float(curr_item.answer)
                    is_correct = abs(user_val - expected_val) < 1e-3
                    correctness_val = 1.0 if is_correct else 0.0
                    resp_time = time.time() - pstate["step_start_time"]

                    prev_b = bkt.get_mean_belief()
                    t_kc = pstate["target_skill_idx"]
                    bkt.update_belief(t_kc, is_correct=is_correct, receives_scaffolding=(curr_act_idx == 4))
                    new_b = bkt.get_mean_belief()

                    # Log interaction to CSV
                    log_interaction(
                        student_id=student_id,
                        tutor_mode=tutor_mode,
                        step=pstate["current_step"] + 1,
                        kc_idx=t_kc,
                        kc_name=KC_NAMES[t_kc],
                        action_idx=curr_act_idx,
                        action_name=curr_act_name,
                        is_correct=is_correct,
                        response_time_sec=resp_time,
                        mean_belief=new_b,
                        beliefs=bkt.get_beliefs().tolist(),
                    )

                    pstate["recent_history"].append(correctness_val)
                    if len(pstate["recent_history"]) > 5:
                        pstate["recent_history"].pop(0)

                    pstate["last_action"] = curr_act_idx
                    pstate["last_friction"] = 0.4 if is_correct else 0.8
                    pstate["current_step"] += 1
                    pstate["last_feedback"] = {
                        "is_correct": is_correct,
                        "correct_ans": int(expected_val) if expected_val.is_integer() else expected_val,
                        "explanation": curr_item.explanation,
                        "delta_b": new_b - prev_b,
                    }

                    # Select Next Action & KC
                    if is_experimental and d3qn_tutor is not None:
                        next_obs = np.array(
                            [bkt.beliefs[0], bkt.beliefs[1], bkt.beliefs[2], bkt.beliefs[3], new_b, float(np.mean(pstate["recent_history"])), pstate["last_friction"], min(pstate["current_step"]/30.0, 1.0), curr_act_idx/4.0],
                            dtype=np.float32,
                        )
                        next_act = d3qn_tutor.select_action(next_obs)
                        next_kc = int(np.argmin(bkt.get_beliefs()))
                    else:  # Control Mode: Linear progression
                        next_kc = min(3, pstate["current_step"] // 5)
                        next_act = (pstate["current_step"] % 3)

                    pstate["current_action"] = next_act
                    pstate["target_skill_idx"] = next_kc
                    pstate["current_item"] = get_curriculum_question(next_kc, next_act, seed=random.randint(1, 10000))
                    pstate["step_start_time"] = time.time()
                    st.rerun()

                except ValueError:
                    st.error("Invalid numerical input! Please enter a valid number (e.g. 12 or 3.5).")
