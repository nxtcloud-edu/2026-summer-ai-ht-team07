/**
 * YEDA 알림 탭 — dry-run 알림 발송
 */

/**
 * 알림 탭 초기화
 */
function initAlert() {
    const btn = document.getElementById("send-alert-btn");
    btn.addEventListener("click", sendAlert);
}

/**
 * 알림 발송 (dry-run)
 */
async function sendAlert() {
    const btn = document.getElementById("send-alert-btn");
    const resultDiv = document.getElementById("alert-result");

    if (!AppState.lastPrediction || !AppState.lastResult) {
        showToast("먼저 예측을 수행하세요.", "error");
        return;
    }

    btn.disabled = true;
    btn.textContent = "발송 중...";

    const body = {
        ...AppState.lastPrediction,
        probability: AppState.lastResult.probability,
        risk_level: AppState.lastResult.risk_level,
    };

    const res = await api.post("/api/alert", body);

    btn.disabled = false;
    btn.textContent = "알림 발송 (Dry-Run)";

    if (!res.ok) {
        resultDiv.classList.remove("hidden");
        resultDiv.textContent = `알림 발송 실패: ${res.error}`;
        resultDiv.style.color = "var(--danger)";
        return;
    }

    const data = res.data;
    resultDiv.classList.remove("hidden");
    resultDiv.style.color = "";

    if (data.body) {
        // dry-run 메일 본문 표시
        resultDiv.textContent = `[Dry-Run 모드 — 실제 발송 안 됨]\n\n제목: ${data.subject || ""}\n수신: ${(data.recipients || []).join(", ") || "(없음)"}\n\n${"─".repeat(50)}\n\n${data.body}`;
    } else {
        resultDiv.textContent = JSON.stringify(data, null, 2);
    }
}
