/**
 * Yield X 예측 탭 — 슬라이더 생성, 프리셋, 예측 호출
 */

// schema.py의 FEATURES를 JS로 정의 (프론트엔드 독립 동작용)
const FEATURES = [
    { name: "uv_time", korean: "UV 조사 시간", unit: "s", low: 3.0, high: 7.0, step: 0.1, adjustable: true, kind: "continuous" },
    { name: "uv_intensity", korean: "UV 광량", unit: "mW/cm\u00B2", low: 1000, high: 1400, step: 10, adjustable: true, kind: "continuous" },
    { name: "pin_speed", korean: "이젝터 핀 상승 속도", unit: "mm/s", low: 0.8, high: 1.6, step: 0.05, adjustable: true, kind: "continuous" },
    { name: "pin_pressure", korean: "핀 압력", unit: "N", low: 20, high: 40, step: 0.5, adjustable: true, kind: "continuous" },
    { name: "pin_height", korean: "핀 상승 높이", unit: "mm", low: 0.3, high: 0.7, step: 0.01, adjustable: true, kind: "continuous" },
    { name: "head_vacuum", korean: "픽업 헤드 진공압", unit: "kPa", low: -70, high: -60, step: 0.5, adjustable: true, kind: "continuous" },
    { name: "pin_vacuum", korean: "이젝터 진공압", unit: "kPa", low: -55, high: -45, step: 0.5, adjustable: true, kind: "continuous" },
    { name: "temperature", korean: "공정 온도", unit: "\u00B0C", low: 20, high: 30, step: 0.5, adjustable: true, kind: "continuous" },
    { name: "humidity", korean: "상대습도", unit: "%RH", low: 30, high: 60, step: 1, adjustable: true, kind: "continuous" },
    { name: "vacuum_status", korean: "진공 시스템 상태", unit: "kPa", low: -100, high: -90, step: 0.5, adjustable: false, kind: "continuous" },
    { name: "runtime_hours", korean: "설비 연속 가동 시간", unit: "h", low: 0, high: 24, step: 0.5, adjustable: false, kind: "continuous" },
    { name: "tape_type", korean: "다이싱 테이프 종류", unit: "", low: 0, high: 1, step: 1, adjustable: false, kind: "categorical" },
    { name: "die_thickness", korean: "다이 두께", unit: "\u03BCm", low: 100, high: 150, step: 1, adjustable: false, kind: "continuous" },
];

// 데모 프리셋 (configs/app.yaml의 demo_presets 미러)
const PRESETS = [
    {
        name: "정상 조건",
        values: { uv_time: 4.8, uv_intensity: 1180, pin_speed: 1.10, pin_pressure: 29.0, pin_height: 0.50, head_vacuum: -65.5, pin_vacuum: -50.0, temperature: 24.0, humidity: 44.0, vacuum_status: -95.5, runtime_hours: 9.0, tape_type: 1, die_thickness: 128.0 }
    },
    {
        name: "위험 — 얇은 다이 + 빠른 핀",
        values: { uv_time: 4.8, uv_intensity: 1180, pin_speed: 1.40, pin_pressure: 29.0, pin_height: 0.50, head_vacuum: -65.5, pin_vacuum: -50.0, temperature: 24.0, humidity: 44.0, vacuum_status: -95.5, runtime_hours: 9.0, tape_type: 1, die_thickness: 112.0 }
    },
    {
        name: "위험 — 마모 + 약한 진공",
        values: { uv_time: 4.8, uv_intensity: 1180, pin_speed: 1.10, pin_pressure: 29.0, pin_height: 0.50, head_vacuum: -62.0, pin_vacuum: -50.0, temperature: 24.0, humidity: 44.0, vacuum_status: -92.0, runtime_hours: 20.0, tape_type: 0, die_thickness: 128.0 }
    }
];

/**
 * 예측 탭 초기화
 */
function initPredict() {
    renderPresetButtons();
    renderSliders();
    setupPredictForm();
    setupCsvUpload();
    setupInputModeToggle();
}

/**
 * 프리셋 버튼 렌더링
 */
function renderPresetButtons() {
    const container = document.getElementById("preset-buttons");
    PRESETS.forEach((preset, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn-preset";
        btn.textContent = preset.name;
        btn.addEventListener("click", () => applyPreset(idx));
        container.appendChild(btn);
    });
}

/**
 * 프리셋 적용
 */
function applyPreset(idx) {
    const values = PRESETS[idx].values;
    FEATURES.forEach(feat => {
        const slider = document.getElementById(`slider-${feat.name}`);
        const display = document.getElementById(`value-${feat.name}`);
        if (slider && values[feat.name] !== undefined) {
            slider.value = values[feat.name];
            display.textContent = formatSliderValue(values[feat.name], feat);
        }
    });
}

/**
 * 슬라이더 값 포맷
 */
function formatSliderValue(val, feat) {
    if (feat.kind === "categorical") {
        return val === 1 ? "UV경화형" : "일반";
    }
    // step에 맞춰 소수점 결정
    const decimals = feat.step < 1 ? Math.max(1, -Math.floor(Math.log10(feat.step))) : 0;
    return Number(val).toFixed(decimals);
}

/**
 * 슬라이더 렌더링
 */
function renderSliders() {
    const container = document.getElementById("sliders-container");
    FEATURES.forEach(feat => {
        const group = document.createElement("div");
        group.className = `slider-group ${feat.adjustable ? "" : "fixed"}`;

        const defaultVal = (feat.low + feat.high) / 2;
        // step 맞춰 스냅
        const snapped = Math.round(defaultVal / feat.step) * feat.step;

        group.innerHTML = `
            <div class="slider-label">
                <span class="name">${feat.korean}${!feat.adjustable ? '<span class="badge-fixed">고정</span>' : ''}</span>
                <span><span class="value" id="value-${feat.name}">${formatSliderValue(snapped, feat)}</span><span class="unit">${feat.unit}</span></span>
            </div>
            <input type="range" id="slider-${feat.name}" 
                   min="${feat.low}" max="${feat.high}" step="${feat.step}" 
                   value="${snapped}">
        `;
        container.appendChild(group);

        // 슬라이더 변경 이벤트
        const slider = group.querySelector("input[type=range]");
        const display = group.querySelector(".value");
        slider.addEventListener("input", () => {
            display.textContent = formatSliderValue(slider.value, feat);
        });
    });
}

/**
 * 현재 슬라이더 값 수집
 */
function collectValues() {
    const values = {};
    FEATURES.forEach(feat => {
        const slider = document.getElementById(`slider-${feat.name}`);
        values[feat.name] = parseFloat(slider.value);
    });
    return values;
}

/**
 * 예측 폼 설정
 */
function setupPredictForm() {
    const form = document.getElementById("predict-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        await runPrediction();
    });
}

/**
 * 예측 실행
 */
async function runPrediction() {
    const values = collectValues();
    AppState.lastPrediction = values;

    const resultDiv = document.getElementById("result-content");
    resultDiv.innerHTML = '<div class="spinner"></div><p style="text-align:center;margin-top:8px;">예측 중...</p>';

    const res = await api.post("/api/predict", values);

    if (!res.ok) {
        resultDiv.innerHTML = `<p style="color:var(--danger);text-align:center;">예측 실패: ${res.error}</p>`;
        showToast(res.error, "error");
        return;
    }

    AppState.lastResult = res.data;
    renderPredictResult(res.data);
}

/**
 * 예측 결과 렌더링
 */
function renderPredictResult(data) {
    const resultDiv = document.getElementById("result-content");
    const prob = (data.probability * 100).toFixed(1);
    const level = data.risk_level || "normal";

    let rangeHtml = "";
    if (data.lower_bound != null && data.upper_bound != null) {
        const lower = (data.lower_bound * 100).toFixed(1);
        const upper = (data.upper_bound * 100).toFixed(1);
        rangeHtml = `<p class="result-range">예상 범위: ${lower}% ~ ${upper}%</p>`;
    }

    resultDiv.innerHTML = `
        <div class="result-card">
            <p class="result-probability ${level}">${prob}%</p>
            ${rangeHtml}
            <span class="result-risk risk-${level}">${getRiskLabel(level)}</span>
            <div class="result-actions">
                <button class="btn-link" onclick="switchTab('explain')">원인 분석 보기</button>
                <button class="btn-link" onclick="switchTab('optimize')">최적화 가이드</button>
            </div>
        </div>
    `;

    // 대시보드에도 반영
    updateDashboardFromResult(data);
    // 알림 버튼 활성화 (위험일 때)
    document.getElementById("send-alert-btn").disabled = (level === "normal");
}

/**
 * 위험 등급 라벨
 */
function getRiskLabel(level) {
    switch (level) {
        case "critical": return "즉시 조치 필요";
        case "warning": return "주의 관찰";
        default: return "정상";
    }
}


// =============================================================================
// CSV 업로드 & 배치 예측
// =============================================================================

let csvData = null; // 파싱된 CSV 행들

/**
 * 입력 모드 토글 설정
 */
function setupInputModeToggle() {
    const manualBtn = document.getElementById("mode-manual");
    const csvBtn = document.getElementById("mode-csv");
    const manualSection = document.getElementById("manual-input-section");
    const csvSection = document.getElementById("csv-input-section");

    manualBtn.addEventListener("click", () => {
        manualBtn.classList.add("active");
        csvBtn.classList.remove("active");
        manualSection.classList.remove("hidden");
        csvSection.classList.add("hidden");
    });

    csvBtn.addEventListener("click", () => {
        csvBtn.classList.add("active");
        manualBtn.classList.remove("active");
        csvSection.classList.remove("hidden");
        manualSection.classList.add("hidden");
    });
}

/**
 * CSV 업로드 설정
 */
function setupCsvUpload() {
    const dropZone = document.getElementById("csv-drop-zone");
    const fileInput = document.getElementById("csv-file-input");
    const predictBtn = document.getElementById("csv-predict-btn");
    const removeBtn = document.getElementById("csv-remove-btn");

    // 클릭으로 파일 선택
    dropZone.addEventListener("click", () => fileInput.click());

    // 파일 선택 이벤트
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleCsvFile(e.target.files[0]);
        }
    });

    // 드래그 앤 드롭
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleCsvFile(e.dataTransfer.files[0]);
        }
    });

    // 배치 예측 버튼
    predictBtn.addEventListener("click", runBatchPrediction);

    // 재학습 버튼
    const retrainBtn = document.getElementById("csv-retrain-btn");
    retrainBtn.addEventListener("click", runRetrain);

    // 모델 초기화 버튼
    const resetBtn = document.getElementById("csv-reset-btn");
    resetBtn.addEventListener("click", runModelReset);

    // 제거 버튼
    removeBtn.addEventListener("click", () => {
        csvData = null;
        fileInput.value = "";
        document.getElementById("csv-file-info").classList.add("hidden");
        predictBtn.disabled = true;
        retrainBtn.disabled = true;
    });
}

/**
 * CSV 파일 처리
 */
function handleCsvFile(file) {
    if (!file.name.endsWith(".csv")) {
        showToast("CSV 파일만 업로드할 수 있습니다.", "error");
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const text = e.target.result;
        const parsed = parseCsv(text);

        if (parsed.error) {
            showToast(parsed.error, "error");
            return;
        }

        csvData = parsed.rows;

        // UI 업데이트 — 결측 건수도 함께 표시
        document.getElementById("csv-file-info").classList.remove("hidden");
        document.getElementById("csv-filename").textContent = file.name;

        let info = `(${csvData.length}행`;
        if (parsed.nullCount > 0) {
            info += `, 결측 ${parsed.nullCount}셀`;
        }
        info += ")";
        document.getElementById("csv-row-count").textContent = info;
        document.getElementById("csv-predict-btn").disabled = false;
        document.getElementById("csv-retrain-btn").disabled = false;
    };
    reader.readAsText(file);
}

/**
 * CSV 파싱 — 결측치(공백, null, NA, NaN 등)를 null로 허용하여 그대로 백엔드에 전달
 *
 * 결측치 처리(imputation)는 백엔드/모델 단에서 수행한다.
 * 프론트는 파싱 + 컬럼 검증 + null 보존만 담당한다.
 */
function parseCsv(text) {
    const lines = text.trim().split("\n");
    if (lines.length < 2) {
        return { error: "CSV에 헤더와 최소 1개 데이터 행이 필요합니다." };
    }

    const header = lines[0].split(",").map(h => h.trim().replace(/"/g, ""));
    const featureNames = FEATURES.map(f => f.name);

    // 필수 컬럼(헤더) 확인 — 컬럼 자체는 있어야 함
    const missing = featureNames.filter(name => !header.includes(name));
    if (missing.length > 0) {
        return { error: `누락된 컬럼: ${missing.join(", ")}` };
    }

    // 결측치로 간주하는 값
    const NULL_VALUES = new Set([
        "", "null", "NULL", "NA", "N/A", "na", "n/a",
        "NaN", "nan", "none", "None", "-", "."
    ]);

    const rows = [];
    let nullCount = 0;

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const values = line.split(",").map(v => v.trim().replace(/"/g, ""));
        const row = {};

        header.forEach((col, idx) => {
            if (featureNames.includes(col)) {
                const raw = idx < values.length ? values[idx].trim() : "";
                if (NULL_VALUES.has(raw)) {
                    // 결측 → null 그대로 백엔드에 전달
                    row[col] = null;
                    nullCount++;
                } else {
                    const num = parseFloat(raw);
                    if (isNaN(num)) {
                        // 숫자로 파싱 불가 → 결측 처리
                        row[col] = null;
                        nullCount++;
                    } else {
                        row[col] = num;
                    }
                }
            }
        });

        rows.push(row);
    }

    if (rows.length === 0) {
        return { error: "유효한 데이터 행이 없습니다." };
    }

    return { rows, nullCount };
}

/**
 * 배치 예측 실행
 */
async function runBatchPrediction() {
    if (!csvData || csvData.length === 0) return;

    const btn = document.getElementById("csv-predict-btn");
    const resultDiv = document.getElementById("result-content");

    btn.disabled = true;
    btn.textContent = `예측 중... (${csvData.length}건)`;
    resultDiv.innerHTML = '<div class="spinner"></div><p style="text-align:center;margin-top:8px;">배치 예측 진행 중...</p>';

    const startTime = performance.now();
    const res = await api.post("/api/predict/batch", { records: csvData });
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    btn.disabled = false;
    btn.textContent = "배치 예측 실행";

    if (!res.ok) {
        resultDiv.innerHTML = `<p style="color:var(--danger);text-align:center;">배치 예측 실패: ${res.error}</p>`;
        showToast(res.error, "error");
        return;
    }

    renderBatchResult(res.data, elapsed);

    // 배치 예측 후 null이 없는 첫 번째 행을 lastPrediction으로 설정 → 원인 분석/가이드 탭 연동
    if (csvData && csvData.length > 0) {
        // null 값이 없는 행을 우선 선택 (SHAP 호출 안정성)
        const cleanRow = csvData.find(row => Object.values(row).every(v => v !== null));
        AppState.lastPrediction = cleanRow || csvData[0];
    }
}

/**
 * 배치 예측 결과 렌더링
 */
function renderBatchResult(data, elapsed = null) {
    const resultDiv = document.getElementById("result-content");
    const results = data.results || [];

    if (results.length === 0) {
        resultDiv.innerHTML = '<p style="text-align:center;color:var(--text-muted);">결과가 없습니다.</p>';
        return;
    }

    // 통계 계산
    const probs = results.map(r => r.probability);
    const avgProb = probs.reduce((a, b) => a + b, 0) / probs.length;
    const minProb = Math.min(...probs);
    const maxProb = Math.max(...probs);
    const criticalCount = results.filter(r => r.risk_level === "critical").length;
    const warningCount = results.filter(r => r.risk_level === "warning").length;

    const timeHtml = elapsed ? `<div class="stat-box"><div class="stat-value">${elapsed}s</div><div class="stat-label">소요 시간</div></div>` : "";

    let html = `
        <div class="batch-summary">
            <div class="stat-box">
                <div class="stat-value">${results.length}</div>
                <div class="stat-label">총 건수</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:var(--primary)">${(avgProb * 100).toFixed(1)}%</div>
                <div class="stat-label">평균 성공률</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${(minProb * 100).toFixed(1)}%</div>
                <div class="stat-label">최솟값</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${(maxProb * 100).toFixed(1)}%</div>
                <div class="stat-label">최댓값</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:var(--danger)">${criticalCount}</div>
                <div class="stat-label">위험(Critical)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:var(--warning)">${warningCount}</div>
                <div class="stat-label">주의(Warning)</div>
            </div>
            ${timeHtml}
        </div>
        <div class="batch-result-wrapper">
            <table class="batch-result-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>예측 성공률</th>
                        <th>예상 범위</th>
                        <th>위험 등급</th>
                    </tr>
                </thead>
                <tbody>
    `;

    results.forEach((r, i) => {
        const prob = (r.probability * 100).toFixed(1);
        const lower = r.lower_bound != null ? (r.lower_bound * 100).toFixed(1) : "—";
        const upper = r.upper_bound != null ? (r.upper_bound * 100).toFixed(1) : "—";
        const levelClass = `risk-${r.risk_level || "normal"}`;
        const levelLabel = getRiskLabel(r.risk_level || "normal");

        html += `<tr>
            <td>${i + 1}</td>
            <td><strong>${prob}%</strong></td>
            <td>${lower}% ~ ${upper}%</td>
            <td><span class="result-risk ${levelClass}" style="font-size:0.75rem;">${levelLabel}</span></td>
        </tr>`;
    });

    html += `</tbody></table></div>`;
    resultDiv.innerHTML = html;

    // 대시보드 업데이트 — Lot 단위 평균 수율로 반영
    const avgResult = {
        probability: avgProb,
        risk_level: avgProb < 0.60 ? "critical" : avgProb < 0.80 ? "warning" : "normal",
        lower_bound: minProb,
        upper_bound: maxProb,
        model_name: results[0]?.model_name || null,
    };
    AppState.lastResult = avgResult;
    updateDashboardFromResult(avgResult);
}

// =============================================================================
// 재학습
// =============================================================================

/**
 * 재학습 실행 — CSV 데이터를 백엔드에 보내 모델을 재학습시킨다
 */
async function runRetrain() {
    if (!csvData || csvData.length === 0) {
        showToast("CSV 데이터를 먼저 업로드하세요.", "error");
        return;
    }

    const btn = document.getElementById("csv-retrain-btn");
    const resultDiv = document.getElementById("retrain-result");

    btn.disabled = true;
    btn.textContent = "재학습 중... (수초 소요)";
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = '<div class="spinner"></div> 모델 재학습 진행 중...';

    const totalStart = performance.now();
    const res = await api.post("/api/retrain", { records: csvData });

    btn.disabled = false;
    btn.textContent = "이 데이터로 재학습";

    if (!res.ok) {
        resultDiv.innerHTML = `<p style="color:var(--danger);">재학습 실패: ${res.error}</p>`;
        showToast("재학습 실패", "error");
        return;
    }

    const data = res.data;
    const metrics = data.metrics || {};
    const retrainElapsed = ((performance.now() - totalStart) / 1000).toFixed(1);

    resultDiv.innerHTML = `
        <p><strong>${data.message}</strong> (총 ${retrainElapsed}초)</p>
        <p>모델: ${data.model_name || "—"} | 추가 데이터: ${data.rows_added}행</p>
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-value">${metrics.accuracy ? (metrics.accuracy * 100).toFixed(1) + "%" : "—"}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${metrics.pr_auc ? (metrics.pr_auc * 100).toFixed(1) + "%" : "—"}</div>
                <div class="metric-label">PR-AUC</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${metrics.recall ? (metrics.recall * 100).toFixed(1) + "%" : "—"}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${metrics.f1_macro ? (metrics.f1_macro * 100).toFixed(1) + "%" : "—"}</div>
                <div class="metric-label">F1-Macro</div>
            </div>
        </div>
    `;

    showToast("재학습 완료! 새 모델이 적용되었습니다.", "info");

    // 서버 상태 배지 업데이트
    const health = await checkHealth();
    if (health.ok) updateServerStatus(health.data);

    // 재학습 후 자동으로 배치 예측 재실행 → 오른쪽에 새 모델 기준 결과 표시
    if (csvData && csvData.length > 0) {
        await runBatchPrediction();
    }
}

// =============================================================================
// 모델 초기화
// =============================================================================

/**
 * 모델을 초기 상태로 되돌린다 (업로드 데이터 삭제 + 원본 데이터로 재학습)
 */
async function runModelReset() {
    const btn = document.getElementById("csv-reset-btn");
    const resultDiv = document.getElementById("result-content");
    const retrainResult = document.getElementById("retrain-result");

    if (!confirm("모델을 초기 상태로 되돌립니다. 업로드된 학습 데이터가 모두 삭제됩니다. 계속할까요?")) {
        return;
    }

    btn.disabled = true;
    btn.textContent = "초기화 중...";

    const res = await api.post("/api/reset-model", {});

    btn.disabled = false;
    btn.textContent = "모델 초기화";

    if (!res.ok) {
        showToast(`초기화 실패: ${res.error}`, "error");
        return;
    }

    showToast("모델이 초기 상태로 복원되었습니다.", "info");

    // UI 초기화
    retrainResult.classList.add("hidden");
    resultDiv.innerHTML = '<p class="result-placeholder">모델이 초기화되었습니다. CSV를 업로드하여 예측하세요.</p>';

    // 서버 상태 업데이트
    const health = await checkHealth();
    if (health.ok) updateServerStatus(health.data);
}
