# rl-edu: Explainable Reinforcement Learning Platform for Intelligent Tutoring Systems

rl-edu is a research-quality, interactive visualization platform that formulates adaptive educational sequencing as a Partially Observable Markov Decision Process (POMDP). Because a student's actual cognitive mastery over skills is hidden, the platform uses Bayesian Knowledge Tracing (BKT) to maintain probabilistic belief distributions over skill masteries. A PyTorch-based Dueling Double Deep Q-Network (D3QN) learns an optimal pedagogical teaching policy that selects the next best instructional action to maximize Normalized Learning Gain (NLG) while minimizing time-on-task and cognitive friction.

---

## Key Features & Visualization Capabilities

The platform provides 15 synchronized interactive visualizations rendered using Plotly and Streamlit:

1. Live BKT Belief Evolution: Real-time multi-line trajectory plot tracking skill mastery probabilities over time with shaded confidence intervals and summary metrics.
2. Interactive Belief Heatmap: Two-dimensional time-step matrix visualizing knowledge component progression using a high-contrast color gradient.
3. D3QN Decision Inspector: Computational flow graph rendering exact tensor values during neural network forward passes: State Input s -> Shared Trunk -> Value Stream V(s) / Advantage Stream A(s,a) -> Q(s,a) Aggregator -> Selected Action.
4. Action Comparison Panel: Horizontal state-action Q-value bar plot highlighting best action, selected action, and runner-up action.
5. Multi-Objective Reward Decomposition: Waterfall chart breaking down learning gain rewards, time-step penalties, hint overuse penalties, and mastery bonus rewards.
6. Hidden State vs. Observable State Comparison: Dual-panel layout comparing surface telemetry (response time, friction, correctness) against true hidden belief states.
7. Student Learning Journey Roadmap: Curriculum progression graph across 4 Knowledge Components with dynamic status nodes (Low, Developing, Mastered).
8. Sequential RL Decision Pipeline: Eight-stage execution timeline tracking data flow from student input to state transition.
9. Policy Evolution & Training Telemetry: Four-panel training dashboard monitoring episode rewards, 50-episode moving average, exploration rate (epsilon decay), action frequency distribution, and Bellman TD loss.
10. Replay Buffer Capacity Inspector: Memory capacity gauge and mini-batch experience tuple inspector.
11. Neural Network Layer Diagram: Feature activation diagram displaying layer node connections and tensor dimensions.
12. Natural Language Explainability Engine: Grounded reasoning card generating data-backed explanations directly from state vectors and Q-values without hallucinating.
13. A/B Benchmark (RL vs. Traditional Tutor): Dual-column split-screen runner comparing PyTorch D3QN policy against standard linear control tutors.
14. Global Session Analytics Dashboard: Supervisor analytics suite evaluating historical interaction data, belief variance, response times, and hint usage.
15. System Architecture Data Flow Animation: Interactive end-to-end flowchart highlighting active components during session interactions.

---

## Repository Architecture

```
rl-edu/
├── app.py                     # Main Streamlit web application dashboard
├── requirements.txt           # Environment dependency specifications
├── README.md                  # Project documentation
├── src/
│   ├── bkt_engine.py          # Multi-skill Bayesian Knowledge Tracing engine
│   ├── pomdp_env.py           # Gymnasium POMDP environment (POMDPTutorEnv)
│   ├── d3qn_agent.py          # PyTorch Dueling Double Deep Q-Network agent
│   ├── curriculum.py          # 4-tier Knowledge Component taxonomy & question repository
│   ├── evaluator.py           # Benchmark evaluation suite vs. baseline policies
│   ├── analytics.py           # Interaction log processing and statistical telemetry
│   ├── logger.py              # Interaction logging (session_logs.csv, affective_surveys.csv)
│   ├── baselines.py           # Random and Heuristic rule-based baseline tutors
│   └── visualization/         # Modular Plotly visualization package
│       ├── __init__.py        # Visualization package initialization
│       ├── bkt_viz.py         # BKT belief trajectory charts and heatmaps
│       ├── d3qn_inspector.py  # D3QN forward pass inspector and action comparison
│       ├── pomdp_explainability.py # Dual-panel inspector, timelines, and explainability engine
│       ├── training_viz.py    # Training telemetry dashboard and policy evolution
│       ├── ab_comparison.py   # A/B split-screen benchmark visualizer
│       └── architecture_viz.py # System architecture animation and global analytics
├── tests/
│   ├── test_env.py            # Pytest suite for BKT math, observation space, and environment steps
│   └── test_visualization.py  # Pytest suite for Plotly visualization generators
├── data/                      # Telemetry log CSV exports
├── models/                    # Trained PyTorch model checkpoints (d3qn_tutor.pt)
└── artifacts/                 # Saved trajectory plots and analysis documents
```

---

## Technical & Mathematical Foundations

### 1. Bayesian Knowledge Tracing (BKT)
Maintains a continuous belief vector b_t over 4 Knowledge Components:
- Basic Arithmetic
- Advanced Arithmetic
- Basic Algebra
- Advanced Algebra

Priors & Parameters:
- Initial Mastery Prior P(L_0) = 0.20
- Learning Transition P(T) = 0.15
- Slip Probability P(S) = 0.10
- Guess Probability P(G) = 0.25
- Scaffolding Boost = 0.10

### 2. Gymnasium POMDP Environment (POMDPTutorEnv)
- Continuous State Space (9D): Skill belief state probabilities b_t, overall mean belief, rolling accuracy, response friction index, normalized step ratio, and normalized last action.
- Discrete Action Space (5 Actions):
  - Action 0: Easy Practice Problem (Difficulty 0.2)
  - Action 1: Medium Practice Problem (Difficulty 0.5)
  - Action 2: Hard Practice Problem (Difficulty 0.8)
  - Action 3: Worked Example Demonstration
  - Action 4: Scaffolding Hint Guidance
- Reward Function:
  Reward = 5.0 * NLG - 0.50_step + Penalty_hint + Bonus_mastery
  where NLG = (b_{t+1} - b_t) / (1.0 - b_t) is the Normalized Learning Gain.

### 3. PyTorch Dueling Double Deep Q-Network (D3QN)
Decouples state value V(s) and action advantage A(s, a):
  Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum_a' A(s, a'))
Uses Double DQN target estimation:
  y_t = r_t + gamma * Q_target(s_{t+1}, argmax_a Q_online(s_{t+1}, a))

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- Virtualenv

### Installation Steps

1. Clone repository:
   ```bash
   git clone https://github.com/Mayur589/rl-edu.git
   cd rl-edu
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run Web Application
Launch the Streamlit interactive visualization dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run Unit Tests
Execute the pytest suite to verify environment dynamics and visualization generators:
```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License.
