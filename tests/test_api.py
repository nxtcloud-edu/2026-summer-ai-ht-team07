"""FastAPI 엔드포인트 테스트.

모델 파일이 없어도 mock 모드로 동작하므로 CI 에서 항상 돌릴 수 있다.

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.ml_service import initialize_service

# 테스트 시작 시 서비스 초기화 (mock 모드로 뜰 것)
initialize_service()

client = TestClient(app)

# 정상 조건 프리셋 (configs/app.yaml 의 "정상 조건" 과 동일)
NORMAL_INPUT = {
    "uv_time": 4.8,
    "uv_intensity": 1180.0,
    "pin_speed": 1.10,
    "pin_pressure": 29.0,
    "pin_height": 0.50,
    "head_vacuum": -65.5,
    "pin_vacuum": -50.0,
    "temperature": 24.0,
    "humidity": 44.0,
    "vacuum_status": -95.5,
    "runtime_hours": 9.0,
    "tape_type": 1.0,
    "die_thickness": 128.0,
}


# ============================================================ Health
class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "is_mock" in data
        assert "model_loaded" in data

    def test_health_mock_mode(self):
        """모델 파일 없이 테스트하므로 mock 모드여야 한다."""
        response = client.get("/api/health")
        data = response.json()
        assert data["is_mock"] is True
        assert data["model_loaded"] is False


# ============================================================ Predict
class TestPredict:
    def test_predict_valid_input(self):
        response = client.post("/api/predict", json=NORMAL_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ("critical", "warning", "normal")

    def test_predict_with_die_id(self):
        payload = {**NORMAL_INPUT, "die_id": "test-001"}
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["die_id"] == "test-001"

    def test_predict_missing_feature(self):
        """피처가 빠지면 422 Validation Error."""
        incomplete = {"uv_time": 5.0, "pin_pressure": 30.0}
        response = client.post("/api/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_out_of_bounds(self):
        """물리적 허용 범위를 벗어나면 422."""
        payload = {**NORMAL_INPUT, "uv_time": 100.0}  # max=7.0
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_bounds(self):
        """음수 범위 (진공압) 검증."""
        payload = {**NORMAL_INPUT, "head_vacuum": -50.0}  # min=-70, max=-60
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422


# ============================================================ Explain
class TestExplain:
    def test_explain_valid_input(self):
        response = client.post("/api/explain", json=NORMAL_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert "base_value" in data
        assert "shap_values" in data
        assert "disclaimer" in data
        assert isinstance(data["shap_values"], list)
        assert len(data["shap_values"]) > 0

    def test_explain_shap_fields(self):
        response = client.post("/api/explain", json=NORMAL_INPUT)
        data = response.json()
        item = data["shap_values"][0]
        assert "feature" in item
        assert "shap_value_pp" in item
        assert "feature_value" in item
        assert item["direction"] in ("기여", "위험")

    def test_explain_with_top_k(self):
        payload = {**NORMAL_INPUT, "top_k": 5}
        response = client.post("/api/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["shap_values"]) <= 5

    def test_explain_disclaimer_present(self):
        response = client.post("/api/explain", json=NORMAL_INPUT)
        data = response.json()
        assert "SHAP" in data["disclaimer"]


# ============================================================ Optimize
class TestOptimize:
    def test_optimize_valid_input(self):
        response = client.post("/api/optimize", json=NORMAL_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert "baseline_prob" in data
        assert "optimized_prob" in data
        assert "gain_pp" in data
        assert "recommendations" in data
        assert "formatted_text" in data
        assert "n_evaluations" in data

    def test_optimize_gain_non_negative(self):
        response = client.post("/api/optimize", json=NORMAL_INPUT)
        data = response.json()
        assert data["gain_pp"] >= -0.01  # 약간의 부동소수 오차 허용

    def test_optimize_recommendations_structure(self):
        response = client.post("/api/optimize", json=NORMAL_INPUT)
        data = response.json()
        if data["recommendations"]:
            item = data["recommendations"][0]
            assert "feature" in item
            assert "current_value" in item
            assert "suggested_value" in item
            assert "delta" in item
            assert "unit" in item
            assert "expected_gain_pp" in item

    def test_optimize_missing_feature(self):
        incomplete = {"uv_time": 5.0}
        response = client.post("/api/optimize", json=incomplete)
        assert response.status_code == 422


# ============================================================ Alert
class TestAlert:
    def test_alert_dry_run(self):
        payload = {
            "die_id": "test-die-001",
            "probability": 0.45,
        }
        response = client.post("/api/alert", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["dry_run"] is True
        assert data["sent"] is False

    def test_alert_normal_condition_no_alert(self):
        """정상 등급이면 알림 조건에 해당하지 않는다."""
        payload = {
            "die_id": "test-die-002",
            "probability": 0.90,  # normal 등급
        }
        response = client.post("/api/alert", json=payload)
        assert response.status_code == 200
        data = response.json()
        # 정상 등급이면 알림 미발송
        assert data["sent"] is False


# ============================================================ Presets
class TestPresets:
    def test_presets_returns_list(self):
        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_preset_structure(self):
        response = client.get("/api/presets")
        data = response.json()
        preset = data[0]
        assert "name" in preset
        assert "description" in preset
        assert "values" in preset
        assert isinstance(preset["values"], dict)

    def test_preset_has_all_features(self):
        """프리셋 값에 13개 피처가 모두 있어야 한다."""
        response = client.get("/api/presets")
        data = response.json()
        from yeda.schema import FEATURE_NAMES

        for preset in data:
            for name in FEATURE_NAMES:
                assert name in preset["values"], f"프리셋 '{preset['name']}'에 {name} 누락"


# ============================================================ History
class TestHistory:
    def test_history_initially_has_entries(self):
        """앞선 predict 테스트에서 이미 이력이 쌓여 있어야 한다."""
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # predict 테스트에서 최소 1건 이상 저장됨
        assert len(data) >= 1

    def test_history_limit_parameter(self):
        response = client.get("/api/history?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 1

    def test_history_item_structure(self):
        response = client.get("/api/history")
        data = response.json()
        if data:
            item = data[0]
            assert "id" in item
            assert "created_at" in item
            assert "probability" in item
            assert "risk_level" in item
            assert "input_json" in item
