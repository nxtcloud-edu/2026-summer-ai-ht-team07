/**
 * YEDA 이력 & 경보 탭 — 수율 추이 + 테이블 + 경보 배너
 */

let historyChart = null;

/**
 * 이력 탭 초기화
 */
function initHistory() {
    // 초기 로드는 탭 진입 시
}

/**
 * 이력 탭 진입 시 호출
 */
async function onHistoryEnter() {
    await loadHistoryWarning();
    await loadHistoryData();
}

/**
 * 경보 상태 로드
 */
async function loadHistoryWarning() {
    const banner = document.getElementById("history-warning-banner");
    const res = await api.get("/api/warning");

    if (!res.ok) {
        banner.classList.add("hidden");
        return;
    }

    const data = res.data;
    if (data.status === "WARNING" || data.status === "CRITICAL") {
        banner.classList.remove("hidden");
        banner.className = `warning-banner ${data.status === "CRITICAL" ? "critical" : ""}`;
        banner.innerHTML = `<strong>\u26A0 수율 저하 경보</strong><br>
            최근 ${data.recent_count || 10}건 평균: ${data.recent_mean != null ? (data.recent_mean * 100).toFixed(1) + "%" : "—"}<br>
            이전 구간 평균: ${data.previous_mean != null ? (data.previous_mean * 100).toFixed(1) + "%" : "—"}<br>
            하락폭: ${data.drop != null ? (data.drop * 100).toFixed(1) + "%p" : "—"}<br>
            ${data.message || ""}`;
    } else {
        banner.classList.add("hidden");
    }
}

/**
 * 이력 데이터 로드 + 차트/테이블 렌더링
 */
async function loadHistoryData() {
    const tbody = document.getElementById("history-tbody");
    const emptyMsg = document.getElementById("history-empty");

    const res = await api.get("/api/history?limit=50");

    // D의 실 응답: 배열 직접 반환, E의 mock: { records: [...] } — 양쪽 호환
    let records = [];
    if (res.ok && res.data) {
        records = Array.isArray(res.data) ? res.data : (res.data.records || []);
    }

    if (records.length === 0) {
        tbody.innerHTML = "";
        emptyMsg.classList.remove("hidden");
        renderHistoryChart([]);
        return;
    }

    emptyMsg.classList.add("hidden");

    // 테이블 렌더링 — timestamp 또는 created_at 호환
    tbody.innerHTML = "";
    records.slice(0, 30).forEach(r => {
        const tr = document.createElement("tr");
        const ts = r.timestamp || r.created_at;
        const time = ts ? new Date(ts).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }) : "—";
        const prob = r.probability != null ? (r.probability * 100).toFixed(1) + "%" : "—";
        const level = r.risk_level || "normal";
        const levelBadge = `<span class="result-risk risk-${level}" style="font-size:0.75rem;">${getRiskLabel(level)}</span>`;
        const model = r.model_name || "—";

        tr.innerHTML = `<td>${time}</td><td>${prob}</td><td>${levelBadge}</td><td>${model}</td>`;
        tbody.appendChild(tr);
    });

    // 차트 렌더링
    renderHistoryChart(records);
}

/**
 * 이력 차트 렌더링
 */
function renderHistoryChart(records) {
    const canvas = document.getElementById("history-chart");
    const ctx = canvas.getContext("2d");

    if (historyChart) {
        historyChart.destroy();
    }

    if (records.length === 0) {
        historyChart = new Chart(ctx, {
            type: "line",
            data: { labels: ["데이터 없음"], datasets: [{ data: [null] }] },
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

    const predicted = records.map(r => r.probability != null ? (r.probability * 100) : null);

    historyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "예측 성공률 (%)",
                data: predicted,
                borderColor: "rgb(37, 99, 235)",
                backgroundColor: "rgba(37, 99, 235, 0.1)",
                fill: true,
                tension: 0.3,
                pointRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "top" }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: "성공률 (%)" }
                }
            }
        }
    });
}
