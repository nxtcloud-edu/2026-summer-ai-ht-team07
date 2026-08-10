/**
 * YEDA API 통신 모듈
 * fetch 래퍼 + 에러 핸들링 + 토스트 알림
 */

// API 베이스 URL — 환경에 따라 자동 전환
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : `${window.location.protocol}//${window.location.hostname}:8000`;

/**
 * API 호출 공통 래퍼
 * @param {string} method - HTTP 메서드
 * @param {string} path - 엔드포인트 경로 (예: "/api/predict")
 * @param {object|null} body - 요청 바디
 * @returns {Promise<{ok: boolean, data?: any, error?: string}>}
 */
async function apiCall(method, path, body = null) {
    const opts = {
        method,
        headers: { "Content-Type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);

    try {
        const res = await fetch(`${API_BASE}${path}`, opts);
        const data = await res.json().catch(() => null);

        if (!res.ok) {
            const msg = data?.detail || data?.error || `HTTP ${res.status}`;
            return { ok: false, error: msg, status: res.status };
        }
        return { ok: true, data };
    } catch (err) {
        return { ok: false, error: "서버에 연결할 수 없습니다.", status: 0 };
    }
}

// 편의 함수
const api = {
    get: (path) => apiCall("GET", path),
    post: (path, body) => apiCall("POST", path, body),
};

/**
 * 서버 상태 확인
 */
async function checkHealth() {
    const res = await api.get("/api/health");
    return res;
}

/**
 * 토스트 메시지 표시
 * @param {string} message - 표시할 메시지
 * @param {string} type - "info" | "error"
 * @param {number} duration - 표시 시간 (ms)
 */
function showToast(message, type = "info", duration = 4000) {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s";
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * 서버 상태 배지 업데이트
 */
function updateServerStatus(healthData) {
    const badge = document.getElementById("server-status");
    if (!healthData) {
        badge.className = "status-badge status-error";
        badge.textContent = "서버 연결 실패";
        return;
    }
    if (healthData.mock_mode) {
        badge.className = "status-badge status-mock";
        badge.textContent = "모의 모드";
    } else {
        badge.className = "status-badge status-ok";
        badge.textContent = "정상 운영";
    }
}
