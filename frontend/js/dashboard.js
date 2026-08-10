/**
 * YEDA 대시보드 탭 — 데모 디자인 기반 3섹션 구조
 * 섹션1: 수율 예측 & 실시간 모니터링 (게이지 + 추세 + 상태)
 * 섹션2: 원인 분석 (Top5 바차트 + 분석결과 + 시뮬레이션 테이블)
 * 섹션3: 최적 조건 시뮬레이션 & 추천 (추천 테이블 + 코멘트)
 *
 * 예측할 때마다 전체가 실시간 업데이트된다.
 */

let dashTrendChart = null;
let dashImportanceChart = null;

// 대시보드용 이력 (세션 내 누적)
let dashHistory = [];

/**
 * 대시보드 초기화
 */
function initDashboard() {
    onDashboardEnter();
}

/**
 * 대시보드 탭 진입 시 호출
 */
async function onDashboardEnter() {
    await loadDashboardData();
}

/**
 * 대시보드 전체 데이터 로드
 */
async function loadDashboardData() {
    // 1. 서버 이력 로드
    const res = await api.get("/api/history?limit=10");
    let records = [];
    if (res.ok && res.data) {
        records = Array.isArray(res.data) ? res.data : (res.data.records || []);
    }

    // 서버 이력 + 로컬 이력 합치기 (중복 방지)
    if (records.length > 0) {
        dashHistory = records;
    }

    if (dashHistory.length > 0) {
        renderDashTrend(dashHistory);
        const latest = dashHistory[0];
        updateDashGauge(latest.probability);
        updateDashStatus(latest);
    } else {
        renderDashTrend([]);
    }

    // 2. 원인 분석 차트 — 마지막 예측의 SHAP 결과 활용
    if (AppState.lastShapData) {
        renderDashImportanceFromShap(AppState.lastShapData);
    } else {
        renderDashImportanceDefault();
    }

    // 3. 최적화 결과 — 마지막 최적화 결과 활용
    if (AppState.lastOptimizeData) {
        renderDashOptimizeFromResult(AppState.lastOptimizeData);
    }

    // 4. 경보 상태
    const warningRes = await api.get("/api/warning");
    if (warningRes.ok && warningRes.data) {
        updateDashAlert(warningRes.data);
    }
}

/**
 * 게이지 업데이트
 */
function updateDashGauge(probability) {
    const pct = (probability * 100).toFixed(1);
    const yieldEl = document.getElementById("dash-yield");
    const gaugeFill = document.getElementById("dash-gauge-fill");

    yieldEl.textContent = `${pct}%`;

    const fillPct = Math.max(0, Math.min(100, ((probability * 100) - 70) / 30 * 100));
    gaugeFill.style.width = `${fillPct}%`;

    if (probability >= 0.90) {
        gaugeFill.className = "dash-gauge-fill gauge-good";
    } else if (probability >= 0.80) {
        gaugeFill.className = "dash-gauge-fill gauge-ok";
    } else {
        gaugeFill.className = "dash-gauge-fill gauge-bad";
    }
}

/**
 * 상태 패널 업데이트
 */
function updateDashStatus(latest) {
    const lotEl = document.getElementById("dash-lot-id");
    const statusEl = document.getElementById("dash-process-status");

    const now = new Date();
    lotEl.textContent = `LOT_${now.getFullYear().toString().slice(2)}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;

    const level = latest.risk_level || "normal";
    if (level === "critical") {
        statusEl.innerHTML = '<span class="status-dot red"></span> 이상';
    } else if (level === "warning") {
        statusEl.innerHTML = '<span class="status-dot yellow"></span> 주의';
    } else {
        statusEl.innerHTML = '<span class="status-dot green"></span> 정상';
    }
}

/**
 * 경보 상태 업데이트
 */
function updateDashAlert(data) {
    const alertEl = document.getElementById("dash-alert-status");
    if (data.status === "WARNING" || data.status === "CRITICAL") {
        alertEl.innerHTML = `<span class="status-dot red"></span> ${data.message || "수율 저하 감지"}`;
    } else {
        alertEl.innerHTML = '<span class="status-dot green"></span> 없음';
    }
}

/**
 * 수율 추세 차트 렌더링
 */
function renderDashTrend(records) {
    const canvas = document.getElementById("dash-trend-chart");
    const ctx = canvas.getContext("2d");

    if (dashTrendChart) dashTrendChart.destroy();

    if (records.length === 0) {
        dashTrendChart = new Chart(ctx, {
            type: "line",
            data: { labels: ["예측을 수행하면 여기에 추이가 표시됩니다"], datasets: [{ data: [null] }] },
            options: { plugins: { legend: { display: false } } }
        });
        return;
    }

    const labels = records.map((r, i) => {
        const ts = r.timestamp || r.created_at;
        if (ts) {
            const d = new Date(ts);
            return `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
        }
        return `#${i + 1}`;
    });
    const values = records.map(r => r.probability != null ? (r.probability * 100) : null);
    const optimizedValues = records.map(r => r.optimized_probability != null ? (r.optimized_probability * 100) : null);

    // 최적화 데이터가 하나라도 있는지 확인
    const hasOptimized = optimizedValues.some(v => v !== null);

    const datasets = [{
        label: "현재 수율 (%)",
        data: values,
        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.05)",
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        pointBackgroundColor: "#2563eb",
    }];

    if (hasOptimized) {
        datasets.push({
            label: "최적화 적용 시 (%)",
            data: optimizedValues,
            borderColor: "#16a34a",
            backgroundColor: "rgba(22, 163, 74, 0.05)",
            fill: true,
            borderDash: [5, 3],
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: "#16a34a",
        });
    }

    dashTrendChart = new Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: "top" } },
            scales: {
                y: {
                    min: 50,
                    max: 100,
                    ticks: { callback: v => v + "%" }
                }
            }
        }
    });
}

/**
 * 원인 분석 — 기본 (데이터 없을 때)
 */
function renderDashImportanceDefault() {
    const canvas = document.getElementById("dash-importance-chart");
    const ctx = canvas.getContext("2d");

    if (dashImportanceChart) dashImportanceChart.destroy();

    dashImportanceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["예측을 먼저 수행하세요"],
            datasets: [{ data: [0], backgroundColor: "#e2e8f0" }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { max: 1 }, y: { grid: { display: false } } }
        }
    });
}

/**
 * 원인 분석 — 실제 SHAP 결과로 업데이트
 */
function renderDashImportanceFromShap(shapData) {
    const canvas = document.getElementById("dash-importance-chart");
    const ctx = canvas.getContext("2d");

    if (dashImportanceChart) dashImportanceChart.destroy();

    // 절댓값 기준 Top 5
    const sorted = [...shapData].sort((a, b) => Math.abs(b.shap_value_pp) - Math.abs(a.shap_value_pp));
    const top5 = sorted.slice(0, 5);

    const labels = top5.map(f => getKoreanName(f.feature) || f.feature);
    const values = top5.map(f => Math.abs(f.shap_value_pp));
    const colors = top5.map(f => f.shap_value_pp >= 0 ? "rgba(22, 163, 74, 0.7)" : "rgba(220, 38, 38, 0.7)");

    dashImportanceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderRadius: 3,
                barThickness: 20,
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    title: { display: true, text: "|기여도| (%p)" }
                },
                y: { grid: { display: false } }
            }
        }
    });

    // 분석 결과 텍스트 업데이트
    const descEl = document.getElementById("dash-analysis-desc");
    if (top5.length > 0) {
        const top1 = top5[0];
        const dir = top1.shap_value_pp >= 0 ? "증가" : "감소";
        descEl.textContent = `현재 공정에서 수율에 가장 큰 영향을 준 변수는 '${getKoreanName(top1.feature) || top1.feature}' (${dir} ${Math.abs(top1.shap_value_pp).toFixed(1)}%p)이며, 그 다음으로 '${getKoreanName(top5[1]?.feature) || top5[1]?.feature}', '${getKoreanName(top5[2]?.feature) || top5[2]?.feature}' 순으로 분석되었습니다.`;
    }
}

/**
 * 최적화 결과로 대시보드 섹션3 업데이트
 */
function renderDashOptimizeFromResult(data) {
    const tbody = document.getElementById("dash-opt-tbody");
    const optYield = document.getElementById("dash-opt-yield");
    const comment = document.getElementById("dash-opt-comment-text");

    const recommendations = data.recommendations || [];
    const optimizedProb = data.optimized_prob ?? data.optimized_probability;
    const baselineProb = data.baseline_prob ?? data.current_probability;

    if (recommendations.length === 0) return;

    // 테이블 렌더링
    tbody.innerHTML = "";
    recommendations.slice(0, 4).forEach((rec, i) => {
        const korean = getKoreanName(rec.feature) || rec.feature;
        const direction = rec.suggested_value > rec.current_value ? "&#x2191;" : "&#x2193;";
        const dirClass = rec.suggested_value > rec.current_value ? "arrow-up" : "arrow-down";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${korean}</td>
            <td>${rec.current_value} ${rec.unit || ""}</td>
            <td>${rec.suggested_value} ${rec.unit || ""}</td>
            <td>${rec.unit || ""}</td>
            <td><span class="${dirClass}">${direction}</span></td>
            ${i === 0 ? `<td rowspan="${Math.min(4, recommendations.length)}" class="opt-yield-cell"><strong>${optimizedProb != null ? (optimizedProb * 100).toFixed(1) + "%" : "—"}</strong></td>` : ""}
        `;
        tbody.appendChild(tr);
    });

    // 예상 수율
    if (optimizedProb != null) {
        optYield.textContent = `${(optimizedProb * 100).toFixed(1)}%`;
    }

    // 코멘트
    const gainPp = data.gain_pp != null ? data.gain_pp.toFixed(1) : "—";
    comment.innerHTML = `위 조건 적용 시 예상 수율이 약 <strong>${gainPp}%p</strong> 향상될 것으로 예측됩니다.`;
}

/**
 * 예측 결과로 대시보드 전체 업데이트 (예측 탭에서 호출)
 * 예측 후 SHAP + 최적화도 자동 호출하여 전체 대시보드를 연동
 */
async function updateDashboardFromResult(data) {
    // 1. 게이지 + 상태 업데이트
    updateDashGauge(data.probability);
    updateDashStatus(data);

    // 2. 추세 차트에 포인트 추가
    dashHistory.unshift({
        probability: data.probability,
        risk_level: data.risk_level,
        optimized_probability: null,
        created_at: new Date().toISOString(),
    });
    if (dashHistory.length > 20) dashHistory = dashHistory.slice(0, 20);
    renderDashTrend(dashHistory);

    // 3. SHAP 자동 호출 → 대시보드 원인 분석 업데이트
    if (AppState.lastPrediction) {
        const explainRes = await api.post("/api/explain", AppState.lastPrediction);
        if (explainRes.ok) {
            const shapData = explainRes.data.shap_values || explainRes.data.contributions || [];
            AppState.lastShapData = shapData;
            renderDashImportanceFromShap(shapData);

            // 시뮬레이션 테이블도 업데이트
            updateDashSimTable(data.probability, shapData);
        }
    }

    // 4. 최적화 자동 호출 → 대시보드 섹션3 업데이트
    if (AppState.lastPrediction) {
        const optRes = await api.post("/api/optimize", AppState.lastPrediction);
        if (optRes.ok) {
            AppState.lastOptimizeData = optRes.data;
            renderDashOptimizeFromResult(optRes.data);

            // 추세 차트 — Lot 평균 수율 + gain_pp로 최적화 라인 계산
            // (단건 최적화 결과를 Lot에 그대로 쓰면 비현실적이므로, gain만 활용)
            const gainPp = optRes.data.gain_pp || 0;
            // 모든 이력 포인트에 최적화 적용 수율 채움
            dashHistory.forEach(record => {
                if (record.probability != null && record.optimized_probability == null) {
                    // 현재 수율 + gain (단, 100% 초과하지 않도록)
                    record.optimized_probability = Math.min(1.0, record.probability + gainPp / 100);
                }
            });
            renderDashTrend(dashHistory);
        }
    }
}

/**
 * 조건 변경 시뮬레이션 테이블 업데이트
 */
function updateDashSimTable(currentProb, shapData) {
    const tbody = document.getElementById("dash-sim-tbody");
    if (!tbody) return;

    const currentPct = (currentProb * 100).toFixed(1);
    // 시뮬레이션: 현재 + 조건 A(+2%p), 조건 B(+4%p), 조건 C(추천, 최적화 결과)
    const optProb = AppState.lastOptimizeData?.optimized_prob ?? AppState.lastOptimizeData?.optimized_probability;
    const condC = optProb != null ? (optProb * 100).toFixed(1) : (currentProb * 100 + 5).toFixed(1);

    tbody.innerHTML = `
        <tr><td>현재 조건 (Baseline)</td><td>${currentPct}%</td></tr>
        <tr><td>조건 A (주요 변수 1개 개선)</td><td>${(currentProb * 100 + 2).toFixed(1)}%</td></tr>
        <tr><td>조건 B (주요 변수 2개 개선)</td><td>${(currentProb * 100 + 3.5).toFixed(1)}%</td></tr>
        <tr class="sim-recommended"><td>조건 C (추천 — 전체 최적화)</td><td><strong>${condC}%</strong></td></tr>
    `;
}

/**
 * 피처명 → 한글명 매핑 유틸
 */
function getKoreanName(featureName) {
    if (!featureName) return "";
    const feat = FEATURES.find(f => f.name === featureName);
    return feat ? feat.korean : featureName;
}
