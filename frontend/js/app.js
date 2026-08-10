/**
 * YEDA 앱 초기화 + 탭 전환 로직
 */

// 전역 상태
const AppState = {
    currentTab: "dashboard",
    lastPrediction: null,   // 마지막 예측 입력값
    lastResult: null,       // 마지막 예측 결과
    healthData: null,
};

/**
 * 탭 전환
 */
function switchTab(tabId) {
    // 탭 버튼 상태
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    // 탭 패널 표시
    document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `tab-${tabId}`);
    });
    AppState.currentTab = tabId;

    // 탭 진입 시 콜백
    if (tabId === "dashboard") onDashboardEnter();
    if (tabId === "history") onHistoryEnter();
    if (tabId === "explain" && AppState.lastPrediction) onExplainEnter();
    if (tabId === "optimize" && AppState.lastPrediction) onOptimizeEnter();
}

/**
 * 앱 초기화
 */
document.addEventListener("DOMContentLoaded", async () => {
    // 탭 버튼 이벤트
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // 서버 상태 확인
    const health = await checkHealth();
    if (health.ok) {
        AppState.healthData = health.data;
        updateServerStatus(health.data);
    } else {
        updateServerStatus(null);
        showToast("백엔드 서버에 연결할 수 없습니다. 일부 기능이 제한됩니다.", "error");
    }

    // 각 탭 모듈 초기화
    initPredict();
    initDashboard();
    initHistory();
    initAlert();

    // 주기적 상태 확인 (30초마다)
    setInterval(async () => {
        const h = await checkHealth();
        if (h.ok) {
            AppState.healthData = h.data;
            updateServerStatus(h.data);
        } else {
            updateServerStatus(null);
        }
    }, 30000);
});
