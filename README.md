# RL Tutor: Explainable Reinforcement Learning for Adaptive Learning Platforms

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6.svg)](https://www.typescriptlang.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A research-grade, production-quality adaptive learning platform powered by an explainable Partially Observable Markov Decision Process (POMDP) and a Dueling Double Deep Q-Network (D3QN) implemented in **pure Python + NumPy**. 

The system implements the **Content Adaptation** and **Guidance Adaptation** framework established in:
> **Riedmann et al. (2025)**. *"Reinforcement Learning in Education: A Systematic Literature Review"*, *International Journal of Artificial Intelligence in Education (IJAIED)*, 35:2669–2723. DOI: [10.1007/s40593-025-00494-6](https://doi.org/10.1007/s40593-025-00494-6).

---

## 1. Pedagogical & Mathematical Foundations

### 1.1 POMDP Formulation
Student knowledge mastery cannot be observed directly; human cognition is latent. The tutoring interaction is formulated as a Partially Observable Markov Decision Process defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$:

1. **Latent State Space $\mathcal{S}$**:
   Unobservable true student mastery vector $L_t^* \in \{0, 1\}^K$ across $K=4$ Knowledge Components (KCs).
2. **Observation State Space $\Omega \subset \mathbb{R}^{10}$**:
   The agent observes a continuous 10-dimensional feature vector $s_t$:
   - $s_t[0:4]$: Continuous Bayesian Knowledge Tracing belief vector $b_t = [P(L_t^{(0)}=1), \dots, P(L_t^{(3)}=1)] \in [0, 1]^4$.
   - $s_t[4]$: Global mean mastery across all KCs: $\bar{b}_t = \frac{1}{K}\sum_{k=0}^{K-1} b_t^{(k)}$.
   - $s_t[5]$: Rolling correctness accuracy over the student's recent 5 attempts.
   - $s_t[6]$: Normalized cognitive friction / response latency index (capturing hesitation and struggle).
   - $s_t[7]$: Normalized current step progress: $t / T_{\max}$.
   - $s_t[8]$: Normalized prior pedagogical action index: $a_{t-1} / (|A| - 1)$.
   - $s_t[9]$: Spaced review urgency index derived from memory retention decay.

3. **Discrete Pedagogical Action Space $\mathcal{A} = \{0, 1, 2, 3, 4, 5\}$**:
   - **Action 0 (Content Adaptation)**: Easy Practice Problem ($\beta = 0.20$)
   - **Action 1 (Content Adaptation)**: Medium Practice Problem ($\beta = 0.50$)
   - **Action 2 (Content Adaptation)**: Hard Practice Problem ($\beta = 0.80$)
   - **Action 3 (Guidance Adaptation)**: Scaffolding Hint (Immediate conceptual cue)
   - **Action 4 (Guidance Adaptation)**: Worked Example Demonstration (Full step-by-step model)
   - **Action 5 (Review Adaptation)**: Spaced Repetition Practice (Targeting decay-prone KCs)

---

### 1.2 Cognitive Modeling: Multi-Skill BKT & Retention Decay

#### Bayesian Knowledge Tracing (BKT)
Belief states update dynamically upon each student response $O_t \in \{0, 1\}$:

$$\begin{aligned}
P(L_t = 1 \mid O_t = 1) &= \frac{b_t (1 - P(S))}{b_t (1 - P(S)) + (1 - b_t) P(G)} \\
P(L_t = 1 \mid O_t = 0) &= \frac{b_t P(S)}{b_t P(S) + (1 - b_t) (1 - P(G))} \\
b_{t+1} &= P(L_t = 1 \mid O_t) + (1 - P(L_t = 1 \mid O_t)) \cdot P(T_{\text{effective}})
\end{aligned}$$

where:
- $P(L_0) = 0.20$ (Prior initial mastery)
- $P(T) = 0.15$ (Learning transition probability)
- $P(S) = 0.10$ (Slip probability: wrong despite mastery)
- $P(G) = 0.25$ (Guess probability: right despite non-mastery)
- $P(T_{\text{effective}}) = P(T) + 0.10$ when scaffolding hints or worked examples are studied.

#### Memory Retention & Spaced Review Urgency
Unpracticed skills experience exponential Ebbinghaus retention decay:
$$b_{\text{decayed}}(t) = b_{\text{floor}} + (b - b_{\text{floor}}) \cdot e^{-\lambda \Delta t}$$
Urgency spikes when previously proficient skills ($b > 0.40$) risk memory decay, giving the RL agent an incentive to schedule Action 5.

---

### 1.3 Multi-Objective Reward Decomposition
The agent optimizes a transparent, decomposed scalar reward:
$$R(s_t, a_t, s_{t+1}) = w_1 \cdot \text{NLG} - C_{\text{step}} - C_{\text{hint}} - C_{\text{friction}} + R_{\text{review}} + R_{\text{mastery}}$$
- **Normalized Learning Gain (NLG)**:
  $$\text{NLG} = \frac{\bar{b}_{t+1} - \bar{b}_t}{1.0 - \bar{b}_t + \epsilon}, \quad R_{\text{NLG}} = 5.0 \times \text{NLG}$$
- **Efficiency Step Cost**: $C_{\text{step}} = -0.40$ (forces rapid mastery and minimizes time-on-task).
- **Hint Over-reliance Penalty**: $C_{\text{hint}} = -0.80$ if consecutive hints $> 2$.
- **Cognitive Friction Penalty**: $C_{\text{friction}} = -0.30 \times \text{friction}$.
- **Spaced Review Bonus**: $R_{\text{review}} = +1.50 \times \text{urgency}$ when review is scheduled appropriately.
- **Terminal Mastery Bonus**: $R_{\text{mastery}} = +50.0$ when all KCs achieve $\ge 88\%$ mastery.

---

### 1.4 Dueling Double Deep Q-Network (D3QN) in Pure NumPy
To eliminate heavyweight runtime dependencies, maximize transparency, and guarantee explainability, the neural network and backpropagation are written from scratch in NumPy:

$$\begin{aligned}
\text{Trunk}: \quad & h_1 = \text{ReLU}(s W_1 + b_1), \quad h_2 = \text{ReLU}(h_1 W_2 + b_2) \\
\text{Value Stream}: \quad & V(s) = \text{ReLU}(h_2 W_{v1} + b_{v1}) W_{v2} + b_{v2} \in \mathbb{R}^1 \\
\text{Advantage Stream}: \quad & A(s, a) = \text{ReLU}(h_2 W_{a1} + b_{a1}) W_{a2} + b_{a2} \in \mathbb{R}^{|A|} \\
\text{Aggregation}: \quad & Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')\right)
\end{aligned}$$

Double DQN Target Estimation:
$$y_t = r_t + \gamma (1 - d_t) \cdot Q_{\text{target}}\left(s_{t+1}, \arg\max_a Q_{\text{online}}(s_{t+1}, a)\right)$$
Optimized with Adam optimizer ($\beta_1 = 0.9, \beta_2 = 0.999$) and Polyak soft target synchronization ($\tau = 0.20$).

---

## 2. Architecture & Technology Stack

| Layer | Technology | Architectural Responsibility |
| :--- | :--- | :--- |
| **Frontend** | React 19 + TypeScript + Tailwind CSS | Interactive student UI, scaffolding drawer, responsive dark glassmorphism. |
| **State** | Zustand | Client state management for auth, session progress, live telemetry. |
| **Charts** | Recharts | Multi-line BKT belief trajectories, D3QN advantage bar charts, reward waterfalls. |
| **Backend** | FastAPI (Python 3.13) | REST API endpoints, session management, async event loops. |
| **RL Engine** | Python + NumPy | D3QN neural net, experience replay buffer, BKT probabilistic engine. |
| **Database** | PostgreSQL / SQLite | Session interaction logs, step telemetry, user profiles, belief history. |
| **ORM** | SQLAlchemy 2.0 | Type-safe declarative database models and transactions. |
| **Authentication**| JWT + Bcrypt | Secure role-based access control (`student` and `researcher`). |
| **Deployment** | Docker & Docker Compose | Containerized multi-stage microservices with PostgreSQL and Nginx. |

---

## 3. Repository Structure

```
rl-edu/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entrypoint, middleware, lifespan
│   │   ├── core/                       # Settings, JWT security, structured logging, exceptions
│   │   ├── domain/                     # Pure domain logic (independent of framework/DB)
│   │   │   ├── curriculum/             # 4-tier KC taxonomy, 84 psychometric questions, DAG
│   │   │   ├── cognitive/              # Multi-skill BKT engine & Ebbinghaus retention decay
│   │   │   └── rl/                     # POMDPTutorEnv, NumPy D3QN, ReplayBuffer, Reward, Baselines
│   │   ├── db/                         # SQLAlchemy 2.0 engine, sessions, and ORM tables
│   │   ├── schemas/                    # Pydantic request/response schemas
│   │   └── api/                        # REST Controllers (/auth, /curriculum, /tutor, /analytics, /benchmarks)
│   ├── tests/                          # 17 Pytest automated test suites
│   ├── requirements.txt                # Python backend dependencies
│   └── Dockerfile                      # Multi-stage Python 3.13 slim container
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Main application container
│   │   ├── api/client.ts               # Axios client with JWT interceptors
│   │   ├── store/useTutorStore.ts      # Zustand global state store
│   │   ├── types/index.ts              # TypeScript domain types
│   │   ├── components/
│   │   │   ├── common/                 # Header, AuthModal
│   │   │   └── student/                # QuestionCard, HintDrawer, WorkedExampleModal, FeedbackModal
│   │   └── pages/
│   │       ├── StudentTutorPage.tsx    # Live adaptive learning interface
│   │       └── ResearcherDashboardPage.tsx # Recharts BKT trajectory, D3QN inspector, A/B benchmarks
│   ├── nginx.conf                      # Nginx SPA & reverse proxy configuration
│   ├── package.json                    # Node dependencies
│   └── Dockerfile                      # Multi-stage Node 20 build & Nginx serving
├── docker-compose.yml                  # Full-stack Docker orchestration
└── README.md                           # Research & architectural documentation
```

---

## 4. Quickstart: Local Development

### Prerequisites
- **Python**: 3.13+ (or 3.11+)
- **Node.js**: 20+ (with npm)

### Step 1: Start Backend
```bash
cd backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Probe: [http://localhost:8000/health](http://localhost:8000/health)

### Step 2: Start Frontend
```bash
cd frontend
npm install
npm run dev
```
- Web Application: [http://localhost:5173](http://localhost:5173)

### Instant One-Click Demo Access
In the web application login screen, click:
- **"Demo Student"**: Launches an interactive learning session with question solving, hints, and feedback.
- **"Demo Researcher"**: Opens the research dashboard with real-time BKT curves, D3QN decision inspectors, and benchmark runners.

---

## 5. Production Deployment with Docker Compose

To build and run the complete containerized stack (PostgreSQL, FastAPI Backend, and React Nginx Frontend) in one command:

```bash
docker-compose up --build -d
```

Services will be online at:
- **Web Application**: `http://localhost:3000`
- **FastAPI REST API**: `http://localhost:8000`
- **PostgreSQL Database**: `localhost:5432`

To shut down:
```bash
docker-compose down
```

---

## 6. Verification & Automated Test Suite

Run the full automated pytest suite covering BKT mathematics, POMDP environment steps, NumPy neural net gradients, and REST API flows:

```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```

All 17 tests validate:
- BKT Bayesian posterior updates and scaffolding boost.
- POMDP 10D continuous state bounds and 6D discrete action transitions.
- NumPy Dueling network decomposition $Q(s, a) = V(s) + A(s, a) - \bar{A}(s)$ and analytical backprop gradients.
- ReplayBuffer uniform sampling and Double DQN target synchronization.
- End-to-end FastAPI authentication and tutoring session workflows.
- Cohen's $d$ effect size and statistical significance testing.

Frontend TypeScript typecheck and production build:
```bash
cd frontend
npm run build
```

---

## 7. A/B Benchmark Results

Monte Carlo simulation across multi-episode student learning trajectories comparing **Explainable D3QN (Ours)** against standard educational baselines:

| Pedagogical Policy | Mean NLG | Cumulative Reward | Cohen's $d$ vs D3QN | Statistically Significant ($p < 0.05$) |
| :--- | :--- | :--- | :--- | :--- |
| **Explainable D3QN (Ours)** | **57.62%** | **-5.060** | *Reference* | — |
| Rule-Based Heuristic | 30.59% | -7.630 | $d = 2.241$ | **Yes** |
| Random Policy | 28.82% | -8.121 | $d = 2.667$ | **Yes** |
| Leitner Spaced Repetition | 15.21% | -7.968 | $d = 3.550$ | **Yes** |
| Static Linear Curriculum | 3.37% | -8.945 | $d = 3.293$ | **Yes** |

The D3QN agent achieves **statistically superior Normalized Learning Gain ($p < 0.001$, Cohen's $d > 2.0$)** by dynamically interleaving practice difficulty, delivering scaffolding hints only when cognitive friction is detected, and scheduling spaced reviews before memory decay occurs.
