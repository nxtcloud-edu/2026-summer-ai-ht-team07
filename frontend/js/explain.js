/**
 * YEDA 원인 분석 탭 — SHAP 기여도 바 차트
 */

let shapChart = null;

/**
 * 원인 분석 탭 진입 시 호출
 */
async function onExplainEnter() {
    if (!AppState.lastPrediction) {
        return;
    }

    const canvas = document.getElementById("shap-chart");
    const summary = document.getElementById("explain-summary");
    const hint = canvas.previousElementSibling;

    // 로딩
    if (hint) hint.textContent = "분석 중...";

    const res = await api.post("/api/explain", AppState.lastPrediction);

    if (!res.ok) {
        if (hint) hint.textContent = `분석 실패: ${res.error}`;
        summary.innerHTML = "";
        return;
    }

    if (hint) hint.classList.add("hidden");

    const contributions = res.data.contributions || res.data;
    renderShapChart(contributions);
    renderExplainSummary(contributions);
}

/**
 * SHAP 바 차트 렌더링
 */
function renderShapChart(contributions) {
    const canvas = document.getElementById("shap-chart");
    const ctx = canvas.getContext("2d");

    // 절댓값 기준 정렬 (영향 큰 순)
    const sorted = [...contributions].sort((a, b) => Math.abs(b.shap_value_pp) - Math.abs(a.shap_value_pp));
    const top = sorted.slice(0, 10); // 상위 10개만

    const labels = top.map(c => c.korean || c.feature);
    const values = top.map(c => c.shap_value_pp);
    const colors = values.map(v => v >= 0 ? "rgba(22, 163, 74, 0.7)" : "rgba(220, 38, 38, 0.7)");
    const borderColors = values.map(v => v >= 0 ? "rgb(22, 163, 74)" : "rgb(220, 38, 38)");

    if (shapChart) {
        shapChart.destroy();
    }

    shapChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "SHAP 기여도 (%p)",
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x >= 0 ? "+" : ""}${ctx.parsed.x.toFixed(1)}%p`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "기여도 (%p)" },
                    grid: { color: "#e2e8f0" }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });

    // 차트 높이 조절
    canvas.style.height = `${Math.max(200, top.length * 35)}px`;
}

/**
 * 해석 요약 렌더링
 */
function renderExplainSummary(contributions) {
    const summary = document.getElementById("explain-summary");
    const sorted = [...contributions].sort((a, b) => Math.abs(b.shap_value_pp) - Math.abs(a.shap_value_pp));

    if (sorted.length === 0) {
        summary.innerHTML = "<p>분석 결과가 없습니다.</p>";
        return;
    }

    const top1 = sorted[0];
    const direction1 = top1.shap_value_pp >= 0 ? "높이고" : "낮추고";

    let text = `현재 조건에서 <strong>${top1.korean || top1.feature}</strong>이(가) 성공률을 가장 크게 ${direction1} 있습니다 (${top1.shap_value_pp >= 0 ? "+" : ""}${top1.shap_value_pp.toFixed(1)}%p).`;

    if (sorted.length > 1) {
        const top2 = sorted[1];
        const direction2 = top2.shap_value_pp >= 0 ? "증가" : "감소";
        text += ` <strong>${top2.korean || top2.feature}</strong>도 ${direction2} 방향으로 ${Math.abs(top2.shap_value_pp).toFixed(1)}%p 기여하고 있습니다.`;
    }

    summary.innerHTML = `<p>${text}</p>
        <p style="margin-top:8px;font-size:0.8rem;color:var(--text-muted);">
        ※ 양수 = 성공률 증가 기여, 음수 = 감소 기여. 기여도는 SHAP 확률 분해 기준입니다.
        </p>`;
}
