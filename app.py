"""Streamlit Web Application for Reinforcement Learning Intelligent Tutoring System.

This module implements an interactive web interface featuring:
1. Simulation Dashboard: Visualizes training performance and benchmarks DQN vs baselines.
2. Live Tutor Mode: Real-time adaptive math problem delivery where the DQN agent selects difficulty.
"""

import os
import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.environment import SimulatedStudentEnv
from src.baselines import RandomTutor, HeuristicTutor
from src.rl_agent import DQNTutor, train_dqn_agent, load_dqn_agent
from src.evaluate import evaluate_tutor


# --- Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="RL Intelligent Tutoring System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .question-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 6px solid #3B82F6;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .feedback-correct {
        background-color: #DCFCE7;
        border: 1px solid #86EFAC;
        color: #166534;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .feedback-incorrect {
        background-color: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper: Math Question Generator ---
def generate_math_question(difficulty: int) -> Tuple[str, float, str]:
    """Generates a math problem based on difficulty level.

    Args:
        difficulty: Difficulty action index (0: Easy, 1: Medium, 2: Hard).

    Returns:
        Tuple[str, float, str]: Question text, numerical answer, explanation text.
    """
    if difficulty == 0:  # Easy: Basic Addition / Subtraction
        op = random.choice(["+", "-"])
        if op == "+":
            a = random.randint(5, 25)
            b = random.randint(3, 20)
            ans = float(a + b)
            q_str = f"What is {a} + {b}?"
            exp = f"{a} + {b} = {int(ans)}"
        else:
            a = random.randint(10, 35)
            b = random.randint(1, a)
            ans = float(a - b)
            q_str = f"What is {a} - {b}?"
            exp = f"{a} - {b} = {int(ans)}"

    elif difficulty == 1:  # Medium: Multiplication / Division / Simple Algebra
        q_type = random.choice(["mult", "div", "alg"])
        if q_type == "mult":
            a = random.randint(4, 12)
            b = random.randint(4, 12)
            ans = float(a * b)
            q_str = f"What is {a} × {b}?"
            exp = f"{a} × {b} = {int(ans)}"
        elif q_type == "div":
            b = random.randint(3, 10)
            ans = float(random.randint(3, 12))
            a = int(ans * b)
            q_str = f"What is {a} ÷ {b}?"
            exp = f"{a} ÷ {b} = {int(ans)}"
        else:  # Alg: ax + b = c
            x_ans = random.randint(2, 9)
            a = random.randint(2, 5)
            b = random.randint(1, 15)
            c = a * x_ans + b
            ans = float(x_ans)
            q_str = f"Solve for x: {a}x + {b} = {c}"
            exp = f"{a}x = {c} - {b} = {c - b} ➔ x = {c - b} ÷ {a} = {int(ans)}"

    else:  # Hard: Exponents / Squares / Multi-step Algebra
        q_type = random.choice(["square", "power", "multi_alg"])
        if q_type == "square":
            base = random.randint(6, 15)
            ans = float(base * base)
            q_str = f"What is {base}² ({base} squared)?"
            exp = f"{base} × {base} = {int(ans)}"
        elif q_type == "power":
            base = random.randint(2, 5)
            exp_val = random.randint(3, 4)
            ans = float(base ** exp_val)
            q_str = f"What is {base}^{exp_val} ({base} to the power of {exp_val})?"
            exp = f"{base}^{exp_val} = {int(ans)}"
        else:  # Multi-step: ax - b = c
            x_ans = random.randint(3, 12)
            a = random.randint(3, 7)
            b = random.randint(5, 20)
            c = a * x_ans - b
            ans = float(x_ans)
            q_str = f"Solve for x: {a}x - {b} = {c}"
            exp = f"{a}x = {c} + {b} = {c + b} ➔ x = {c + b} ÷ {a} = {int(ans)}"

    return q_str, ans, exp


# --- Helper: Load or Cache Model ---
@st.cache_resource
def get_cached_dqn_model(model_path: str = "models/dqn_tutor.zip"):
    """Loads and caches the trained DQN model.

    Args:
        model_path: Path to model zip file.

    Returns:
        Loaded DQN model instance or None if file not found.
    """
    if os.path.exists(model_path):
        env = SimulatedStudentEnv(max_steps=30)
        return load_dqn_agent(model_path, env=env)
    return None


# --- App Header ---
st.markdown('<div class="main-header">🎓 Intelligent Tutoring System (RL-ITS)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Reinforcement Learning-based Adaptive Question Scheduling powered by Deep Q-Networks (DQN)</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation & Settings")
app_mode = st.sidebar.radio(
    "Select Interface Mode:",
    ["📊 Simulation Dashboard", "🎓 Live Tutor Mode"],
)

model_path = "models/dqn_tutor.zip"
dqn_model = get_cached_dqn_model(model_path)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")
if dqn_model is not None:
    st.sidebar.success("✅ Trained DQN Model Loaded")
else:
    st.sidebar.warning("⚠️ Model Not Found (`models/dqn_tutor.zip`)")
    if st.sidebar.button("🔨 Train DQN Model Now"):
        with st.spinner("Training DQN Agent (30,000 steps)..."):
            env = SimulatedStudentEnv(max_steps=30)
            train_dqn_agent(env, total_timesteps=30000, seed=42, model_save_path=model_path)
            st.cache_resource.clear()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**MDP Formulation:**\n"
    "- **State Space:** Knowledge level θ, recent accuracy history, difficulty streak.\n"
    "- **Action Space:** Question Difficulty (Easy = 0.2, Medium = 0.5, Hard = 0.8).\n"
    "- **Reward:** Learning gain Δθ + ZPD alignment bonus."
)


# ==============================================================================
# MODE 1: SIMULATION DASHBOARD
# ==============================================================================
if app_mode == "📊 Simulation Dashboard":
    st.header("📊 Benchmark & Policy Evaluation Dashboard")
    st.write(
        "Compare the adaptive performance of the **DQN RL Agent** against baseline tutoring policies "
        "(Random Tutor and Heuristic Rule-Based Tutor)."
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        eval_episodes = st.slider("Evaluation Episodes per Policy:", min_value=10, max_value=100, value=30, step=10)
    with col_ctrl2:
        eval_seed = st.number_input("Random Seed:", min_value=1, max_value=9999, value=42, step=1)
    with col_ctrl3:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Full Benchmark", type="primary", use_container_width=True)

    if run_btn or "benchmark_results" in st.session_state:
        if run_btn or "benchmark_results" not in st.session_state:
            with st.spinner("Running simulation benchmark across policies..."):
                env = SimulatedStudentEnv(max_steps=30)
                random_tutor = RandomTutor(env.action_space, seed=eval_seed)
                heuristic_tutor = HeuristicTutor(initial_difficulty=1)

                if dqn_model is None:
                    train_dqn_agent(env, total_timesteps=30000, seed=eval_seed, model_save_path=model_path)
                    st.cache_resource.clear()
                    dqn_model = get_cached_dqn_model(model_path)

                dqn_tutor = DQNTutor(dqn_model, deterministic=True)

                rand_res = evaluate_tutor(random_tutor, env, n_episodes=eval_episodes, seed=eval_seed)
                heur_res = evaluate_tutor(heuristic_tutor, env, n_episodes=eval_episodes, seed=eval_seed)
                dqn_res = evaluate_tutor(dqn_tutor, env, n_episodes=eval_episodes, seed=eval_seed)

                st.session_state["benchmark_results"] = {
                    "Random Tutor": rand_res,
                    "Heuristic Tutor": heur_res,
                    "DQN Tutor (RL)": dqn_res,
                }

        results = st.session_state["benchmark_results"]

        # --- Benchmark Metrics Comparison Cards ---
        st.subheader("🏆 Policy Benchmark Summary")
        m_col1, m_col2, m_col3 = st.columns(3)

        for col, (policy_name, res) in zip([m_col1, m_col2, m_col3], results.items()):
            with col:
                st.markdown(f"### {policy_name}")
                st.metric("Mean Cumulative Reward", f"{res['mean_reward']:.2f}", delta=f"± {res['std_reward']:.2f}")
                st.metric("Final Student Knowledge (θ)", f"{res['mean_final_knowledge']:.3f}", delta=f"± {res['std_final_knowledge']:.3f}")
                st.metric("Average Answer Accuracy", f"{res['mean_accuracy'] * 100:.1f}%")
                st.metric("Mastery Rate (θ ≥ 0.95)", f"{res['mastery_rate'] * 100:.1f}%")

        st.markdown("---")

        # --- Trajectory Line Chart ---
        st.subheader("📈 Student Knowledge Progression Trajectory (Average across Episodes)")

        # Compute average knowledge state at each step (0 to 30) for each policy
        max_len = 31
        traj_df = pd.DataFrame({"Step": list(range(max_len))})

        for policy_name, res in results.items():
            trajs = res["knowledge_trajectories"]
            padded_trajs = []
            for t in trajs:
                # pad or slice to max_len
                if len(t) < max_len:
                    t = t + [t[-1]] * (max_len - len(t))
                padded_trajs.append(t[:max_len])
            mean_traj = np.mean(padded_trajs, axis=0)
            traj_df[policy_name] = mean_traj

        traj_chart_data = traj_df.set_index("Step")
        st.line_chart(traj_chart_data, use_container_width=True)

        # --- Difficulty Action Distribution Bar Chart ---
        st.subheader("🎯 Difficulty Action Choice Distribution")
        diff_counts = {}
        diff_names = {0: "Easy (0.2)", 1: "Medium (0.5)", 2: "Hard (0.8)"}

        for policy_name, res in results.items():
            all_actions = [a for episode_actions in res["difficulty_trajectories"] for a in episode_actions]
            total_actions = max(1, len(all_actions))
            diff_counts[policy_name] = {
                diff_names[a_idx]: (all_actions.count(a_idx) / total_actions) * 100 for a_idx in [0, 1, 2]
            }

        diff_df = pd.DataFrame(diff_counts)
        st.bar_chart(diff_df, use_container_width=True)


# ==============================================================================
# MODE 2: LIVE TUTOR MODE (INTERACTIVE HUMAN PRACTICE)
# ==============================================================================
elif app_mode == "🎓 Live Tutor Mode":
    st.header("🎓 Interactive Live Math Tutor")
    st.write(
        "Experience the Intelligent Tutor in real-time! As you answer math questions, the loaded **DQN RL Agent** "
        "evaluates your state and dynamically adapts the problem difficulty."
    )

    if dqn_model is None:
        st.error("⚠️ Trained DQN Model (`models/dqn_tutor.zip`) is missing. Please train the model from the sidebar first.")
        st.stop()

    # --- Session State Initialization ---
    if "tutor_state" not in st.session_state:
        st.session_state["tutor_state"] = {
            "student_knowledge": 0.20,
            "last_difficulty": 0.0,
            "last_correctness": 0.0,
            "recent_history": [],
            "consecutive_correct": 0,
            "consecutive_incorrect": 0,
            "current_step": 0,
            "max_steps": 20,
            "score": 0,
            "current_action": 1,  # Start at Medium
            "current_question": None,
            "current_answer": None,
            "current_explanation": None,
            "last_feedback": None,
        }
        # Generate first question
        q_str, ans, exp = generate_math_question(1)
        st.session_state["tutor_state"]["current_question"] = q_str
        st.session_state["tutor_state"]["current_answer"] = ans
        st.session_state["tutor_state"]["current_explanation"] = exp

    state = st.session_state["tutor_state"]

    # Top Control Bar & Reset Button
    col_reset1, col_reset2 = st.columns([5, 1])
    with col_reset2:
        if st.button("🔄 Restart Practice Session"):
            del st.session_state["tutor_state"]
            st.rerun()

    # Progress & Knowledge Dashboard Header
    st.markdown("---")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("Questions Attempted", f"{state['current_step']} / {state['max_steps']}")

    with col_stat2:
        st.metric("Estimated Knowledge (θ)", f"{state['student_knowledge']:.2f}")

    with col_stat3:
        st.metric("Current Score", f"{state['score']}")

    with col_stat4:
        diff_labels = {0: "🟢 Easy", 1: "🟡 Medium", 2: "🔴 Hard"}
        st.metric("DQN Next Difficulty", diff_labels[state["current_action"]])

    # Knowledge Progress Bar
    st.write("**Student Mastery Level Progress:**")
    st.progress(float(min(1.0, max(0.0, state["student_knowledge"]))))

    # Check if session completed
    if state["student_knowledge"] >= 0.95 or state["current_step"] >= state["max_steps"]:
        st.balloons()
        st.success(
            f"🎉 **Practice Session Complete!**\n\n"
            f"- **Final Knowledge Level:** {state['student_knowledge']:.3f}\n"
            f"- **Total Score:** {state['score']} / {state['current_step']}\n"
            f"- **Mastery Status:** {'Achieved (θ ≥ 0.95)!' if state['student_knowledge'] >= 0.95 else 'Completed Max Steps'}"
        )
        if st.button("Start New Session", type="primary"):
            del st.session_state["tutor_state"]
            st.rerun()
        st.stop()

    # --- Last Question Feedback Banner ---
    if state["last_feedback"] is not None:
        fb = state["last_feedback"]
        if fb["is_correct"]:
            st.markdown(
                f'<div class="feedback-correct">✅ <b>Correct!</b> Excellent work. '
                f'Explanation: <i>{fb["explanation"]}</i> | Knowledge gained: <b>+{fb["delta_k"]:.3f}</b></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="feedback-incorrect">❌ <b>Incorrect.</b> Your answer was {fb["user_ans"]}. '
                f'Correct answer: <b>{fb["correct_ans"]}</b>. Explanation: <i>{fb["explanation"]}</i></div>',
                unsafe_allow_html=True,
            )
        st.write("")

    # --- Present Current Question ---
    st.markdown(
        f'<div class="question-box">'
        f'<h3>Question {state["current_step"] + 1} (Difficulty: {diff_labels[state["current_action"]]})</h3>'
        f'<h2 style="color: #1E40AF;">{state["current_question"]}</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Answer Input Form with Validation
    with st.form(key="answer_form", clear_on_submit=True):
        user_input_str = st.text_input("Enter your numerical answer (e.g. 12, 3.5):", key="user_answer_input")
        submit_btn = st.form_submit_button("Submit Answer", type="primary")

    if submit_btn:
        cleaned_input = user_input_str.strip()
        if not cleaned_input:
            st.warning("Please enter an answer before submitting.")
        else:
            try:
                user_val = float(cleaned_input)
                expected_val = float(state["current_answer"])
                is_correct = abs(user_val - expected_val) < 1e-3
                correctness_val = 1.0 if is_correct else 0.0

                diff_val_map = {0: 0.2, 1: 0.5, 2: 0.8}
                d_val = diff_val_map[state["current_action"]]

                # Update Student Knowledge State in session
                prev_k = state["student_knowledge"]
                if is_correct:
                    learning_gain = 0.15 * (1.0 - prev_k) * d_val
                    new_k = min(1.0, prev_k + learning_gain)
                    state["consecutive_correct"] += 1
                    state["consecutive_incorrect"] = 0
                    state["score"] += 1
                else:
                    attempt_gain = 0.02 * (1.0 - prev_k)
                    new_k = min(1.0, prev_k + attempt_gain)
                    state["consecutive_incorrect"] += 1
                    state["consecutive_correct"] = 0

                state["student_knowledge"] = float(new_k)
                delta_k = new_k - prev_k

                # Update history
                state["recent_history"].append(correctness_val)
                if len(state["recent_history"]) > 5:
                    state["recent_history"].pop(0)

                state["last_difficulty"] = d_val
                state["last_correctness"] = correctness_val
                state["current_step"] += 1

                # Save feedback for rendering
                state["last_feedback"] = {
                    "is_correct": is_correct,
                    "user_ans": cleaned_input,
                    "correct_ans": int(expected_val) if expected_val.is_integer() else expected_val,
                    "explanation": state["current_explanation"],
                    "delta_k": delta_k,
                }

                # Construct Observation Vector for DQN Agent
                rolling_acc = float(np.mean(state["recent_history"])) if state["recent_history"] else 0.0
                norm_correct = min(state["consecutive_correct"] / 5.0, 1.0)
                norm_incorrect = min(state["consecutive_incorrect"] / 5.0, 1.0)
                norm_step = min(state["current_step"] / float(state["max_steps"]), 1.0)

                obs = np.array(
                    [
                        state["student_knowledge"],
                        state["last_difficulty"],
                        state["last_correctness"],
                        rolling_acc,
                        norm_correct,
                        norm_incorrect,
                        norm_step,
                    ],
                    dtype=np.float32,
                )

                # Predict next question difficulty using loaded DQN model!
                action_pred, _ = dqn_model.predict(obs, deterministic=True)
                next_action = int(action_pred)
                state["current_action"] = next_action

                # Generate Next Question
                next_q, next_ans, next_exp = generate_math_question(next_action)
                state["current_question"] = next_q
                state["current_answer"] = next_ans
                state["current_explanation"] = next_exp

                st.rerun()

            except ValueError:
                st.error("Invalid input! Please enter a valid number (e.g. 12 or 4.5).")
