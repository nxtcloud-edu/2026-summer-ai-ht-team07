/**
 * YEDA 대시보드 탭 — 데모 디자인 기반 3섹션 구조
 * 섹션1: 수율 예측 & 실시간 모니터링 (게이지 + 추세 + 상태)
 * 섹션2: 원인 분석 (Top5 바차트 + 분석결과 + 시뮬레이션 테이블)
 * 섹션3: 최적 조건 시뮬레이션 & 추천 (추천 테이블 + 코멘트)
 */

let dashTrendChart = null;
let dashImportanceChart = null;

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
    // 1. 최근 이력으로 추세 차트 + 현재 수율 업데이트
    const historyRes = await api.get("/api/history?limit=10");
    if (historyRes.ok && historyRes.data && Array.isArray(historyRes.data.records) && historyRes.data.records.length > 0) {
        const records = historyRes.data.records;
        renderDashTrend(records);
        // 가장 최근 예측으로 게이지 업데이트
        const latest = records[records.length - 1];
        updateDashGauge(latest.probability);
        updateDashStatus(latest);
    } else {
        renderDashTrend([]);
    }

    // 2. 원인 분석 차트 (mock explain 활용)
    renderDashImportance();

    // 3. 경보 상태
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

    // 게이지 범위: 70~100%, 스케일 변환
    const fillPct = Math.max(0, Math.min(100, ((probability * 100) - 70) / 30 * 100));
    gaugeFill.style.width = `${fillPct}%`;

    // 색상
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

    // Lot ID (mock)
    const now = new Date();
    lotEl.textContent = `LOT_${now.getFullYear().toString().slice(2)}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;

    // 공정 상태
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
            data: { labels: ["데이터 없음"], datasets: [{ data: [null] }] },
            options: { plugins: { legend: { display: false } } }
        });
        return;
    }

    const labels = records.map((r, i) => `Lot-${records.length - i}`);
    const values = records.map(r => r.probability != null ? (r.probability * 100) : null);

    dashTrendChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "예상 수율 (%)",
                data: values,
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.05)",
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointBackgroundColor: "#2563eb",
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    min: 70,
                    max: 100,
                    ticks: { callback: v => v + "%" }
                }
            }
        }
    });
}

/**
 * 원인 분석 Top 5 바 차트
 */
function renderDashImportance() {
    const canvas = document.getElementById("dash-importance-chart");
    const ctx = canvas.getContext("2d");

    if (dashImportanceChart) dashImportanceChart.destroy();

    // Mock 데이터 (실제로는 /api/explain에서 가져옴)
    const features = [
        { name: "Pin pressure", value: 0.42 },
        { name: "Head vacuum", value: 0.31 },
        { name: "Pin height", value: 0.18 },
        { name: "UV intensity", value: 0.06 },
        { name: "Temperature", value: 0.03 },
    ];

    dashImportanceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: features.map(f => f.name),
            datasets: [{
                data: features.map(f => f.value),
                backgroundColor: "#2563eb",
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
                    min: 0,
                    max: 0.6,
                    title: { display: true, text: "영향도 (중요도)" }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * 예측 결과로 대시보드 전체 업데이트 (예측 탭에서 호출)
 */
function updateDashboardFromResult(data) {
    updateDashGauge(data.probability);

    // Lot ID 갱신
    const lotEl = document.getElementById("dash-lot-id");
    const now = new Date();
    lotEl.textContent = `LOT_${now.getFullYear().toString().slice(2)}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;

    // 상태
    const statusEl = document.getElementById("dash-process-status");
    const level = data.risk_level || "normal";
    if (level === "critical") {
        statusEl.innerHTML = '<span class="status-dot red"></span> 이상';
    } else if (level === "warning") {
        statusEl.innerHTML = '<span class="status-dot yellow"></span> 주의';
    } else {
        statusEl.innerHTML = '<span class="status-dot green"></span> 정상';
    }

    // 최적화 예상 수율 업데이트
    const optYield = document.getElementById("dash-opt-yield");
    const improvedPct = Math.min(99.9, data.probability * 100 + 4.9);
    optYield.textContent = `${improvedPct.toFixed(1)}%`;

    const comment = document.getElementById("dash-opt-comment-text");
    comment.innerHTML = `위 조건 적용 시 예상 수율이 약 <strong>4.9%</strong> 향상될 것으로 예측됩니다.`;
}
