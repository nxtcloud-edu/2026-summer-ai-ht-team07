/**
 * YEDA 가이드 탭 — 최적화 제안 표시
 */

/**
 * 가이드 탭 진입 시 호출
 */
async function onOptimizeEnter() {
    if (!AppState.lastPrediction) {
        return;
    }

    const summary = document.getElementById("optimize-summary");
    const table = document.getElementById("optimize-table");
    const tbody = document.getElementById("optimize-tbody");
    const limitations = document.getElementById("optimize-limitations");
    const hint = table.previousElementSibling;

    summary.innerHTML = '<div class="spinner"></div> 최적화 계산 중...';
    table.classList.add("hidden");
    limitations.innerHTML = "";

    const res = await api.post("/api/optimize", AppState.lastPrediction);

    if (!res.ok) {
        summary.innerHTML = `<p style="color:var(--danger);">최적화 실패: ${res.error}</p>`;
        return;
    }

    if (hint) hint.classList.add("hidden");

    const data = res.data;
    renderOptimizeResult(data);
}

/**
 * 최적화 결과 렌더링
 */
function renderOptimizeResult(data) {
    const summary = document.getElementById("optimize-summary");
    const table = document.getElementById("optimize-table");
    const tbody = document.getElementById("optimize-tbody");
    const limitations = document.getElementById("optimize-limitations");

    // 요약
    const currentPct = data.current_probability != null ? (data.current_probability * 100).toFixed(1) : "—";
    const optimizedPct = data.optimized_probability != null ? (data.optimized_probability * 100).toFixed(1) : "—";
    const gainPp = data.gain_pp != null ? data.gain_pp.toFixed(1) : "—";

    summary.innerHTML = `
        <p><strong>현재 예측 성공률:</strong> ${currentPct}%</p>
        <p><strong>최적화 후 예상:</strong> ${optimizedPct}% <span style="color:var(--success);">(+${gainPp}%p)</span></p>
    `;

    // 제안 테이블
    const recommendations = data.recommendations || [];
    if (recommendations.length > 0) {
        tbody.innerHTML = "";
        recommendations.forEach(rec => {
            const delta = rec.suggested_value - rec.current_value;
            const deltaStr = (delta >= 0 ? "+" : "") + delta.toFixed(2);
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${rec.korean || rec.feature}</td>
                <td>${rec.current_value}${rec.unit ? " " + rec.unit : ""}</td>
                <td><strong>${rec.suggested_value}</strong>${rec.unit ? " " + rec.unit : ""}</td>
                <td>${deltaStr}</td>
                <td style="color:var(--success);">+${rec.expected_gain_pp.toFixed(1)}%p</td>
            `;
            tbody.appendChild(tr);
        });
        table.classList.remove("hidden");
    } else {
        table.classList.add("hidden");
    }

    // 한계/제약 표시
    const lims = data.limitations || [];
    if (lims.length > 0) {
        limitations.innerHTML = "<strong>제약 사항:</strong><ul>" +
            lims.map(l => `<li>${l}</li>`).join("") +
            "</ul><p style='margin-top:8px;'>※ 조정 불가 변수(제품 사양·설비 상태)는 제안 대상에서 제외됩니다.</p>";
    } else {
        limitations.innerHTML = "<p>※ 조정 불가 변수(다이 두께, 테이프 종류 등)는 제안 대상에서 제외됩니다.</p>";
    }
}
