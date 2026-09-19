"""Integration tests for FastAPI REST Endpoints using TestClient."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_curriculum_endpoints():
    response = client.get("/api/v1/curriculum/kcs")
    assert response.status_code == 200
    assert len(response.json()) == 4

    response = client.get("/api/v1/curriculum/questions?kc_idx=0&difficulty=Easy")
    assert response.status_code == 200
    assert len(response.json()) == 7


def test_full_tutoring_session_flow():
    # 1. Register Student
    uname = "pytest_student_1"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": f"{uname}@test.edu", "password": "password123"},
    )

    # 2. Login
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username": uname, "password": "password123"},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Start Tutoring Session
    start_res = client.post(
        "/api/v1/tutor/sessions/start",
        json={"policy_type": "d3qn"},
        headers=headers,
    )
    assert start_res.status_code == 200
    session_id = start_res.json()["session_id"]

    # 4. Fetch Next Action & Question
    action_res = client.get(
        f"/api/v1/tutor/sessions/{session_id}/next-action",
        headers=headers,
    )
    assert action_res.status_code == 200
    action_data = action_res.json()
    assert "question" in action_data
    assert "explainability" in action_data
    q_id = action_data["question"]["id"]

    # 5. Submit Answer
    ans_res = client.post(
        f"/api/v1/tutor/sessions/{session_id}/submit-answer",
        json={
            "question_id": q_id,
            "student_answer": 12.0,
            "response_time_seconds": 3.2,
            "hint_viewed": False,
            "worked_example_viewed": False,
        },
        headers=headers,
    )
    assert ans_res.status_code == 200
    ans_data = ans_res.json()
    assert "is_correct" in ans_data
    assert "reward_breakdown" in ans_data

    # 6. Retrieve Session Analytics
    analytics_res = client.get(
        f"/api/v1/analytics/sessions/{session_id}",
        headers=headers,
    )
    assert analytics_res.status_code == 200
    analytics = analytics_res.json()
    assert len(analytics["belief_trajectory"]) >= 2
    assert len(analytics["reward_timeline"]) >= 1
