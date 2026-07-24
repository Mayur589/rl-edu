"""Advanced Streamlit Web Application for POMDP-BKT Intelligent Tutoring System.

This application provides:
1. POMDP-BKT Simulation Dashboard: Benchmarks PyTorch D3QN agent vs baselines with NLG metrics.
2. Live Interactive Tutor & D3QN Explainability: Real-time BKT multi-skill belief tracking
   and neural network Q-value / Advantage decomposition for AI decision explainability.
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Force PyTorch single-threading to prevent Streamlit watcher thread segmentation faults on macOS
torch.set_num_threads(1)

from src.bkt_engine import BKTEngine
from src.pomdp_env import POMDPTutorEnv
from src.baselines import RandomTutor
from src.d3qn_agent import D3QNAgent, D3QNTutor, train_d3qn
from src.evaluator import POMDPHeuristicTutor, evaluate_pomdp_tutor, run_pomdp_benchmark


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
        agent = D3QNAgent(state_dim=8, action_dim=5, device="cpu")
        agent.load(model_path)
        agent.online_net.eval()
        return agent
    return None


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="POMDP-BKT Intelligent Tutoring System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .skill-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .question-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 6px solid #2563EB;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .worked-example-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 6px solid #D97706;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #78350F;
    }
    .hint-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #166534;
    }
    .explain-card {
        background-color: #F1F5F9;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #CBD5E1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Problem & Scaffolding Generator ---
def generate_pomdp_content(action_idx: int, skill_idx: int) -> Tuple[str, float, str, str]:
    """Generates problem text, answer, explanation, and hint based on action and skill.

    Args:
        action_idx: Action index in {0, 1, 2, 3, 4}.
        skill_idx: Target skill index in {0, 1, 2}.

    Returns:
        Tuple[str, float, str, str]: (Prompt, Answer, Explanation, Hint Text).
    """
    hint_text = ""

    if skill_idx == 0:  # Skill 0: Addition & Subtraction
        if action_idx == 0:  # Easy
            a, b = random.randint(5, 20), random.randint(3, 15)
            ans = float(a + b)
            prompt = f"What is {a} + {b}?"
            exp = f"{a} + {b} = {int(ans)}"
        elif action_idx == 1:  # Medium
            a, b = random.randint(25, 60), random.randint(15, 45)
            ans = float(a - b)
            prompt = f"What is {a} - {b}?"
            exp = f"{a} - {b} = {int(ans)}"
        elif action_idx == 2:  # Hard
            a, b, c = random.randint(15, 40), random.randint(12, 30), random.randint(5, 20)
            ans = float(a + b - c)
            prompt = f"Calculate: {a} + {b} - {c}"
            exp = f"{a} + {b} = {a+b}, then {a+b} - {c} = {int(ans)}"
        elif action_idx == 3:  # Worked Example
            a, b = 38, 27
            ans = 65.0
            prompt = f"💡 Worked Example: How to add multi-digit numbers like {a} + {b}"
            exp = f"Step 1: Add units place (8 + 7 = 15).\nStep 2: Carry 1 to tens place (3 + 2 + 1 = 6).\nResult = {int(ans)}"
        else:  # Action 4: Scaffolding Hint
            a, b = random.randint(18, 45), random.randint(12, 35)
            ans = float(a + b)
            prompt = f"Solve with Hint: What is {a} + {b}?"
            hint_text = f"💡 Hint: Break {b} into ({b//10*10} + {b%10}). First add {a} + {b//10*10} = {a + b//10*10}."
            exp = f"{a} + {b} = {int(ans)}"

    elif skill_idx == 1:  # Skill 1: Multiplication & Division
        if action_idx == 0:  # Easy
            a, b = random.randint(3, 9), random.randint(2, 6)
            ans = float(a * b)
            prompt = f"What is {a} × {b}?"
            exp = f"{a} × {b} = {int(ans)}"
        elif action_idx == 1:  # Medium
            b = random.randint(4, 9)
            ans = float(random.randint(4, 12))
            a = int(ans * b)
            prompt = f"What is {a} ÷ {b}?"
            exp = f"{a} ÷ {b} = {int(ans)}"
        elif action_idx == 2:  # Hard
            a, b = random.randint(11, 16), random.randint(6, 12)
            ans = float(a * b)
            prompt = f"Calculate: {a} × {b}"
            exp = f"{a} × {b} = {int(ans)}"
        elif action_idx == 3:  # Worked Example
            a, b = 48, 6
            ans = 8.0
            prompt = f"💡 Worked Example: How to solve division problems like {a} ÷ {b}"
            exp = f"Step 1: Find what number multiplied by {b} equals {a}.\nSince {b} × 8 = {a}, {a} ÷ {b} = {int(ans)}."
        else:  # Action 4: Scaffolding Hint
            a, b = random.randint(6, 12), random.randint(4, 8)
            ans = float(a * b)
            prompt = f"Solve with Hint: What is {a} × {b}?"
            hint_text = f"💡 Hint: {a} × {b} is the same as adding {a} to itself {b} times."
            exp = f"{a} × {b} = {int(ans)}"

    else:  # Skill 2: Algebra & Exponents
        if action_idx == 0:  # Easy
            base = random.randint(3, 8)
            ans = float(base * base)
            prompt = f"What is {base}² ({base} squared)?"
            exp = f"{base} × {base} = {int(ans)}"
        elif action_idx == 1:  # Medium
            x_ans = random.randint(2, 8)
            a, b = random.randint(2, 5), random.randint(3, 12)
            c = a * x_ans + b
            ans = float(x_ans)
            prompt = f"Solve for x: {a}x + {b} = {c}"
            exp = f"{a}x = {c} - {b} = {c - b} ➔ x = {c - b} ÷ {a} = {int(ans)}"
        elif action_idx == 2:  # Hard
            base, p = random.randint(2, 4), random.randint(3, 4)
            ans = float(base ** p)
            prompt = f"What is {base}^{p} ({base} to the power of {p})?"
            exp = f"{base}^{p} = {int(ans)}"
        elif action_idx == 3:  # Worked Example
            prompt = "💡 Worked Example: Solving linear equation 3x + 5 = 20"
            ans = 5.0
            exp = "Step 1: Subtract 5 from both sides: 3x = 15.\nStep 2: Divide by 3: x = 5."
        else:  # Action 4: Scaffolding Hint
            x_ans = random.randint(3, 9)
            a, b = random.randint(2, 4), random.randint(4, 15)
            c = a * x_ans - b
            ans = float(x_ans)
            prompt = f"Solve with Hint: {a}x - {b} = {c}"
            hint_text = f"💡 Hint: First add {b} to both sides to isolate {a}x."
            exp = f"{a}x = {c + b} ➔ x = {int(ans)}"

    return prompt, ans, exp, hint_text


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
        agent = D3QNAgent(state_dim=8, action_dim=5)
        agent.load(model_path)
        return agent
    return None


# --- App Header ---
st.markdown('<div class="main-header">🎓 Research-Grade POMDP Intelligent Tutoring System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Bayesian Knowledge Tracing (BKT) + PyTorch Dueling Double Deep Q-Network (D3QN) Explainability Engine</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation & Settings")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["📊 POMDP-BKT Simulation Dashboard", "🎓 Live Interactive Tutor & Explainability"],
)

model_path = "models/d3qn_tutor.pt"
d3qn_agent = load_d3qn_cached(model_path)

st.sidebar.markdown("---")
st.sidebar.subheader("PyTorch D3QN Model")
if d3qn_agent is not None:
    st.sidebar.success("✅ PyTorch D3QN Model Loaded (`models/d3qn_tutor.pt`)")
else:
    st.sidebar.warning("⚠️ Model Not Found")
    if st.sidebar.button("🔨 Train D3QN Agent Now"):
        with st.spinner("Training PyTorch D3QN Agent (500 episodes)..."):
            env = POMDPTutorEnv(max_steps=30)
            train_d3qn(env, total_episodes=500, save_path=model_path, seed=42)
            st.cache_resource.clear()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**POMDP-BKT Architecture:**\n"
    "- **Belief State (b_t):** P(L_t) across 3 skills.\n"
    "- **Actions (5):** Easy/Med/Hard Problem, Worked Example, Scaffolding Hint.\n"
    "- **Reward:** Normalized Learning Gain (NLG) + time-on-task penalty."
)


# ==============================================================================
# MODE 1: POMDP-BKT SIMULATION DASHBOARD
# ==============================================================================
if app_mode == "📊 POMDP-BKT Simulation Dashboard":
    st.header("📊 Comparative Policy Evaluation Dashboard")
    st.write(
        "Evaluate the **PyTorch Dueling Double DQN (D3QN)** policy against baseline tutors on the POMDP environment."
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        eval_episodes = st.slider("Evaluation Episodes per Policy:", min_value=10, max_value=100, value=30, step=10)
    with c2:
        eval_seed = st.number_input("Random Seed:", min_value=1, max_value=9999, value=42, step=1)
    with c3:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Benchmark Evaluation", type="primary", use_container_width=True)

    if run_btn or "pomdp_results" in st.session_state:
        if run_btn or "pomdp_results" not in st.session_state:
            with st.spinner("Running POMDP-BKT evaluation benchmark..."):
                if d3qn_agent is None:
                    env = POMDPTutorEnv(max_steps=30)
                    train_d3qn(env, total_episodes=500, save_path=model_path, seed=eval_seed)
                    st.cache_resource.clear()
                    d3qn_agent = load_d3qn_cached(model_path)

                df, raw = run_pomdp_benchmark(n_episodes=eval_episodes, seed=eval_seed, model_path=model_path)
                st.session_state["pomdp_results"] = (df, raw)

        df, results = st.session_state["pomdp_results"]

        # Metric summary cards
        st.subheader("🏆 Policy Performance Summary")
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
        st.subheader("📈 Multi-Skill Belief Trajectory Progression")
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
        st.subheader("🖼️ Generated Analysis Artifacts")
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            if os.path.exists("artifacts/pomdp_belief_trajectories.png"):
                st.image("artifacts/pomdp_belief_trajectories.png", caption="Belief Trajectories Comparison")
        with img_c2:
            if os.path.exists("artifacts/pomdp_nlg_comparison.png"):
                st.image("artifacts/pomdp_nlg_comparison.png", caption="Reward vs. NLG Comparison")


# ==============================================================================
# MODE 2: LIVE INTERACTIVE TUTOR & D3QN EXPLAINABILITY
# ==============================================================================
elif app_mode == "🎓 Live Interactive Tutor & Explainability":
    st.header("🎓 Interactive Live Math Practice & D3QN AI Explainability")
    st.write(
        "Practice math while the **Bayesian Knowledge Tracing (BKT)** engine updates your skill beliefs in real-time, "
        "and inspect the **D3QN Neural Network's** internal State Value $V(s)$ and Advantage $A(s, a)$ decomposition."
    )

    if d3qn_agent is None:
        st.error("⚠️ Trained D3QN Model (`models/d3qn_tutor.pt`) is missing. Please train the model from the sidebar first.")
        st.stop()

    d3qn_tutor = D3QNTutor(d3qn_agent)

    # Initialize Session State
    if "pomdp_state" not in st.session_state:
        bkt = BKTEngine(seed=42)
        init_beliefs = bkt.reset()
        st.session_state["pomdp_state"] = {
            "bkt_engine": bkt,
            "recent_history": [],
            "last_action": 0,
            "last_correctness": 0.0,
            "last_friction": 0.0,
            "current_step": 0,
            "max_steps": 20,
            "score": 0,
            "current_action": 0,
            "target_skill_idx": 0,
            "current_prompt": None,
            "current_answer": None,
            "current_explanation": None,
            "current_hint": None,
            "last_feedback": None,
        }
        # First action prediction
        obs = np.array([init_beliefs[0], init_beliefs[1], init_beliefs[2], float(np.mean(init_beliefs)), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        act = d3qn_tutor.select_action(obs)
        st.session_state["pomdp_state"]["current_action"] = act
        p, a, e, h = generate_pomdp_content(act, 0)
        st.session_state["pomdp_state"]["current_prompt"] = p
        st.session_state["pomdp_state"]["current_answer"] = a
        st.session_state["pomdp_state"]["current_explanation"] = e
        st.session_state["pomdp_state"]["current_hint"] = h

    pstate = st.session_state["pomdp_state"]
    bkt = pstate["bkt_engine"]
    beliefs = bkt.get_beliefs()
    mean_b = bkt.get_mean_belief()

    # Reset Button
    col_hdr1, col_hdr2 = st.columns([5, 1])
    with col_hdr2:
        if st.button("🔄 Restart Practice"):
            del st.session_state["pomdp_state"]
            st.rerun()

    st.markdown("---")

    # --- Live Multi-Skill BKT Belief Dashboard ---
    st.subheader("🧠 Live Bayesian Knowledge Tracing (BKT) Skill Beliefs")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    skill_names = bkt.skills
    with b_col1:
        st.metric(f"➕ {skill_names[0]}", f"{beliefs[0]:.2f}")
        st.progress(float(beliefs[0]))
    with b_col2:
        st.metric(f"✖️ {skill_names[1]}", f"{beliefs[1]:.2f}")
        st.progress(float(beliefs[1]))
    with b_col3:
        st.metric(f"📐 {skill_names[2]}", f"{beliefs[2]:.2f}")
        st.progress(float(beliefs[2]))
    with b_col4:
        st.metric("🌟 Overall Mean Belief (b_t)", f"{mean_b:.2f}")
        st.progress(float(mean_b))

    st.markdown("---")

    # Construct Current State Vector for Explainability
    rolling_acc = float(np.mean(pstate["recent_history"])) if pstate["recent_history"] else 0.0
    norm_step = min(pstate["current_step"] / float(pstate["max_steps"]), 1.0)
    norm_last_act = pstate["last_action"] / 4.0

    current_obs = np.array(
        [beliefs[0], beliefs[1], beliefs[2], mean_b, rolling_acc, pstate["last_friction"], norm_step, norm_last_act],
        dtype=np.float32,
    )

    # Decompose D3QN values
    v_s, adv_list, q_list = d3qn_tutor.predict_explainability(current_obs)

    # --- D3QN Neural Network Explainability Panel ---
    with st.expander("🔍 **D3QN Neural Network Explainability & Action Q-Values**", expanded=True):
        ex_col1, ex_col2 = st.columns([1, 2])
        with ex_col1:
            st.markdown(f"#### State Value V(s)")
            st.metric("V(s) Expected Return", f"{v_s:.3f}")
            st.caption("Estimated cumulative learning return from current student state.")

        with ex_col2:
            st.markdown("#### Action Q-Values & Advantage A(s, a) Breakdown")
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
    if mean_b < 0.90:
        st.info(f"🎯 **Mastery Target:** Reaching Mean Skill Belief $\\bar{{b}}_t \\ge 0.90$ | Current: **{mean_b:.2f}** ({min(100.0, (mean_b/0.90)*100):.1f}% of goal)")

    # Check Mastery Achievement
    if mean_b >= 0.90 or pstate["current_step"] >= pstate["max_steps"]:
        st.balloons()
        st.success(
            f"🏆 **EXCELLENT WORK! ABSOLUTE MASTERY ACHIEVED!** 🏆\n\n"
            f"- **Final Mean Skill Belief:** **{mean_b:.3f}** (Target ≥ 0.90)\n"
            f"- **Steps to Mastery:** **{pstate['current_step']} steps**\n"
            f"- **Mastery Status:** {'✅ Achieved Absolute Mastery (b_t ≥ 0.90)!' if mean_b >= 0.90 else '⏱️ Completed Max Steps'}"
        )

        if st.button("Start New Practice Session", type="primary"):
            del st.session_state["pomdp_state"]
            st.rerun()
        st.stop()


    # Feedback Banner
    if pstate["last_feedback"] is not None:
        fb = pstate["last_feedback"]
        if fb["is_correct"]:
            st.success(f"✅ **Correct!** Explanation: *{fb['explanation']}* | Skill Belief Gain: **+{fb['delta_b']:.3f}**")
        else:
            st.error(f"❌ **Incorrect.** Correct answer: **{fb['correct_ans']}**. Explanation: *{fb['explanation']}*")

    # Present Action Content (Worked Example vs Problem vs Hint)
    st.markdown(f"### Current Step {pstate['current_step'] + 1} | Selected Action: **{curr_act_name}**")

    if curr_act_idx == 3:  # Worked Example Action
        st.markdown(
            f'<div class="worked-example-box">'
            f'<h2>{pstate["current_prompt"]}</h2>'
            f'<p>{pstate["current_explanation"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("I have studied this example ➔ Continue", type="primary"):
            # Update BKT with worked example benefit
            prev_b = bkt.get_mean_belief()
            t_skill = pstate["target_skill_idx"]
            bkt.update_belief(t_skill, is_correct=True, receives_scaffolding=True)
            new_b = bkt.get_mean_belief()

            pstate["current_step"] += 1
            pstate["last_action"] = 3
            pstate["last_friction"] = 0.2
            pstate["last_feedback"] = {
                "is_correct": True,
                "correct_ans": "Studied Example",
                "explanation": pstate["current_explanation"],
                "delta_b": new_b - prev_b,
            }

            # Predict next action
            next_obs = np.array(
                [bkt.beliefs[0], bkt.beliefs[1], bkt.beliefs[2], new_b, 1.0, 0.2, min(pstate["current_step"]/30.0, 1.0), 3.0/4.0],
                dtype=np.float32,
            )
            next_act = d3qn_tutor.select_action(next_obs)
            next_target_skill = int(np.argmin(bkt.get_beliefs()))
            pstate["current_action"] = next_act
            pstate["target_skill_idx"] = next_target_skill

            p, a, e, h = generate_pomdp_content(next_act, next_target_skill)
            pstate["current_prompt"] = p
            pstate["current_answer"] = a
            pstate["current_explanation"] = e
            pstate["current_hint"] = h
            st.rerun()

    else:  # Problem Actions (0, 1, 2) or Hint Action (4)
        if curr_act_idx == 4 and pstate["current_hint"]:
            st.markdown(
                f'<div class="hint-box"><h4>{pstate["current_hint"]}</h4></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="question-box"><h2>{pstate["current_prompt"]}</h2></div>',
            unsafe_allow_html=True,
        )

        with st.form(key="pomdp_answer_form", clear_on_submit=True):
            user_input_str = st.text_input("Enter your numerical answer:", key="pomdp_ans_input")
            submit_btn = st.form_submit_button("Submit Answer", type="primary")

        if submit_btn:
            cleaned = user_input_str.strip()
            if not cleaned:
                st.warning("Please enter a response.")
            else:
                try:
                    user_val = float(cleaned)
                    expected_val = float(pstate["current_answer"])
                    is_correct = abs(user_val - expected_val) < 1e-3
                    correctness_val = 1.0 if is_correct else 0.0

                    prev_b = bkt.get_mean_belief()
                    t_skill = pstate["target_skill_idx"]
                    bkt.update_belief(t_skill, is_correct=is_correct, receives_scaffolding=(curr_act_idx == 4))
                    new_b = bkt.get_mean_belief()

                    pstate["recent_history"].append(correctness_val)
                    if len(pstate["recent_history"]) > 5:
                        pstate["recent_history"].pop(0)

                    pstate["last_action"] = curr_act_idx
                    pstate["last_friction"] = 0.4 if is_correct else 0.8
                    pstate["current_step"] += 1
                    pstate["last_feedback"] = {
                        "is_correct": is_correct,
                        "correct_ans": int(expected_val) if expected_val.is_integer() else expected_val,
                        "explanation": pstate["current_explanation"],
                        "delta_b": new_b - prev_b,
                    }

                    # Predict Next Action
                    next_obs = np.array(
                        [bkt.beliefs[0], bkt.beliefs[1], bkt.beliefs[2], new_b, float(np.mean(pstate["recent_history"])), pstate["last_friction"], min(pstate["current_step"]/30.0, 1.0), curr_act_idx/4.0],
                        dtype=np.float32,
                    )
                    next_act = d3qn_tutor.select_action(next_obs)
                    next_target_skill = int(np.argmin(bkt.get_beliefs()))
                    pstate["current_action"] = next_act
                    pstate["target_skill_idx"] = next_target_skill

                    p, a, e, h = generate_pomdp_content(next_act, next_target_skill)
                    pstate["current_prompt"] = p
                    pstate["current_answer"] = a
                    pstate["current_explanation"] = e
                    pstate["current_hint"] = h
                    st.rerun()

                except ValueError:
                    st.error("Invalid numerical input! Please enter a valid number.")
