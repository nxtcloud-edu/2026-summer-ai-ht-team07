"""YEDA Streamlit 데모.

실행::

    streamlit run app/streamlit_app.py

화면 구성 (핵심 기능 4종에 1:1 대응)
    1. 수율 예측      — 조건 입력 → 예측 픽업 성공률
    2. 원인 분해      — SHAP 워터폴 (확률 %p 단위)
    3. 개선 가이드    — 조정 가능 변수만 탐색한 조건 제안
    4. 알림          — 임계값 초과 시 이메일 (기본 dry-run)

**설계 제약 1순위: 발표 데모가 반드시 돌아간다.**
    - 모델이 없어도 mock 모드로 뜬다.
    - 어떤 탭에서 예외가 나도 해당 탭만 오류를 표시하고 앱은 살아 있는다.
    - 슬라이더를 손으로 맞추다 실수하지 않도록 프리셋 버튼을 둔다.

소유자: E(UI·통합). **이 파일은 E만 수정한다.**
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit 은 실행 위치가 프로젝트 루트가 아닐 수 있어 import 경로를 직접 붙인다.
ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT / "app")):
    if path not in sys.path:
        sys.path.insert(0, path)

from components.backend import get_backend  # noqa: E402
from components.mock_backend import MOCK_BADGE  # noqa: E402
from yeda.alerts.email import build_message, risk_level  # noqa: E402
from yeda.explain.shap_explainer import DISCLAIMER  # noqa: E402
from yeda.io_utils import load_config  # noqa: E402
from yeda.schema import ADJUSTABLE, FEATURE_NAMES, FIXED, SPEC_BY_NAME  # noqa: E402

st.set_page_config(page_title="YEDA", page_icon="🔧", layout="wide")


@st.cache_resource
def _backend():
    """백엔드를 세션 간 재사용한다 (SHAP explainer 재생성 방지)."""
    return get_backend()


@st.cache_data
def _config() -> dict:
    return load_config("app")


def _sidebar_inputs(cfg: dict) -> dict[str, float]:
    """사이드바에 공정 조건 입력 위젯을 그린다.

    조정 가능 변수와 고정 변수를 **시각적으로 분리**한다. 이 구분이 화면에 보이지
    않으면 심사위원이 "제품 사양도 바꾸라는 거냐"고 묻게 된다.

    Args:
        cfg: ``configs/app.yaml``.

    Returns:
        13개 피처 값 dict.
    """
    st.sidebar.header("공정 조건 입력")

    presets = {p["name"]: p for p in cfg.get("demo_presets", [])}
    choice = st.sidebar.selectbox("데모 프리셋", ["(직접 입력)"] + list(presets))
    defaults = presets[choice]["values"] if choice in presets else {}
    if choice in presets:
        st.sidebar.caption(presets[choice]["description"])

    _apply_preset(choice, defaults)

    values: dict[str, float] = {}

    st.sidebar.subheader("설비 파라미터 (조정 가능)")
    for name in ADJUSTABLE:
        values[name] = _widget(name, defaults, key_prefix="adj")

    st.sidebar.subheader("고정 조건 (제품 사양 · 자재 · 설비 상태)")
    st.sidebar.caption("최적화 탐색에서 제외됩니다.")
    for name in FIXED:
        values[name] = _widget(name, defaults, key_prefix="fix")

    return {name: values[name] for name in FEATURE_NAMES}


def _apply_preset(choice: str, defaults: dict[str, float]) -> None:
    """프리셋 선택이 바뀌면 위젯 값을 실제로 갈아끼운다.

    Streamlit 위젯은 ``key`` 가 있으면 첫 렌더 이후 ``value=`` 인자를 무시하고
    session_state 값을 유지한다. 그래서 프리셋만 바꿔서는 슬라이더가 움직이지 않는다.
    session_state 를 **위젯 생성 전에** 직접 덮어써야 반영된다.

    발표 중 슬라이더 13개를 손으로 맞추는 사고를 막는 것이 프리셋의 존재 이유이므로,
    이 함수가 동작하지 않으면 프리셋 기능 자체가 무의미하다.

    Args:
        choice: 선택된 프리셋 이름.
        defaults: 해당 프리셋의 ``{피처명: 값}``. 직접 입력이면 빈 dict.
    """
    if choice == st.session_state.get("_active_preset"):
        return
    st.session_state["_active_preset"] = choice

    for name, value in defaults.items():
        if name not in SPEC_BY_NAME:
            continue
        prefix = "adj" if name in ADJUSTABLE else "fix"
        st.session_state[f"{prefix}_{name}"] = float(value)


def _widget(name: str, defaults: dict, key_prefix: str) -> float:
    """변수 하나에 대한 입력 위젯."""
    spec = SPEC_BY_NAME[name]
    key = f"{key_prefix}_{name}"
    default = float(defaults.get(name, (spec.low + spec.high) / 2))

    # session_state 에 이미 값이 있으면 default 를 넘기지 않는다.
    # 둘 다 주면 Streamlit 이 경고를 띄우고, 어느 쪽이 이기는지 헷갈리게 된다.
    has_state = key in st.session_state

    if spec.kind == "categorical":
        labels = {0.0: "0 · 일반 점착", 1.0: "1 · UV 경화형"}
        kwargs = {} if has_state else {"index": int(default)}
        selected = st.sidebar.radio(
            spec.korean,
            options=[0.0, 1.0],
            format_func=lambda v: labels[v],
            key=key,
            horizontal=True,
            **kwargs,
        )
        return float(selected)

    kwargs = {} if has_state else {"value": min(max(default, spec.low), spec.high)}
    return float(
        st.sidebar.slider(
            f"{spec.korean} ({spec.unit})",
            min_value=float(spec.low),
            max_value=float(spec.high),
            step=float(spec.step),
            key=key,
            help=spec.rationale,
            **kwargs,
        )
    )


def _render_predict(backend, values: dict[str, float], cfg: dict) -> float:
    """탭 1 — 수율 예측."""
    probability = backend.predict_one(values)
    level = risk_level(probability, cfg)
    colors = {"critical": "🔴 즉시 조치", "warning": "🟡 주의 관찰", "normal": "🟢 정상"}

    left, mid, right = st.columns(3)
    left.metric("예측 픽업 성공률", f"{probability * 100:.1f}%")
    mid.metric("위험 등급", colors[level])
    if backend.metrics:
        right.metric("모델 PR-AUC (홀드아웃)", f"{backend.metrics.get('pr_auc', float('nan')):.3f}")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "표시된 성공률은 학습 모델의 예측 확률입니다. 확률을 수율(%)로 해석할 수 있도록 "
        "신뢰도 곡선(calibration)으로 검증했습니다 — `artifacts/metrics/calibration_curve.csv`"
    )
    return probability


def _render_explain(backend, values: dict[str, float]) -> list[dict]:
    """탭 2 — SHAP 원인 분해."""
    frame = backend.to_frame(values)
    explanation = backend.explain(frame, ids=["현재조건"], top_k=8)

    chart = explanation.copy()
    chart["라벨"] = chart["feature"].map(lambda f: SPEC_BY_NAME[f].korean)
    chart = chart.set_index("라벨")[["shap_value_pp"]].rename(
        columns={"shap_value_pp": "성공률 기여 (%p)"}
    )
    st.bar_chart(chart, horizontal=True)

    st.dataframe(
        explanation.assign(
            변수=lambda d: d["feature"].map(lambda f: SPEC_BY_NAME[f].korean),
            현재값=lambda d: d["feature_value"],
            기여도_pp=lambda d: d["shap_value_pp"].round(2),
        )[["변수", "현재값", "기여도_pp", "direction"]],
        hide_index=True,
        width="stretch",
    )
    st.info(DISCLAIMER)

    risky = explanation[explanation["shap_value_pp"] < 0].head(3)
    return [
        {
            "feature": row["feature"],
            "shap_value_pp": row["shap_value_pp"],
            "feature_value": row["feature_value"],
            "korean": SPEC_BY_NAME[row["feature"]].korean,
        }
        for _, row in risky.iterrows()
    ]


def _render_optimize(backend, values: dict[str, float]) -> list[str]:
    """탭 3 — 개선 가이드."""
    result = backend.recommend(values)

    left, right = st.columns(2)
    left.metric("현재 예측 성공률", f"{result.baseline_prob * 100:.1f}%")
    right.metric(
        "제안 조건 예측 성공률",
        f"{result.optimized_prob * 100:.1f}%",
        delta=f"{result.gain_pp:+.1f}%p",
    )

    if result.recommendations.empty:
        st.success("현재 조건에서 유의미한 개선안을 찾지 못했습니다. 이미 양호한 설정입니다.")
        return []

    table = result.recommendations.copy()
    table["변수"] = table["feature"].map(lambda f: SPEC_BY_NAME[f].korean)
    st.dataframe(
        table[["변수", "current_value", "suggested_value", "unit", "expected_gain_pp"]].rename(
            columns={
                "current_value": "현재값",
                "suggested_value": "제안값",
                "unit": "단위",
                "expected_gain_pp": "기대 개선 (%p)",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "제품 사양(다이 두께)·자재(테이프 종류)·설비 상태(가동 시간, 진공 시스템)는 "
        "탐색 대상에서 제외되며, 제안값은 물리적 허용 범위와 설비 설정 분해능으로 제한됩니다."
    )

    from yeda.optimize.search import format_recommendations

    lines = format_recommendations(result) if not backend.is_mock else []
    for line in lines:
        st.write(f"- {line}")
    return lines


def _render_alert(values: dict[str, float], probability: float, risky, guides, cfg: dict) -> None:
    """탭 4 — 이메일 알림."""
    level = risk_level(probability, cfg)
    message = build_message("DEMO-0001", probability, risky, guides, cfg)

    st.write(f"현재 위험 등급: **{level}** / 발송 기준: **{cfg['alerts']['trigger_level']}**")
    if level != cfg["alerts"]["trigger_level"]:
        st.info(
            "현재 조건은 발송 기준에 해당하지 않습니다. 아래는 **본문 미리보기**이며, "
            "실제 운영에서는 발송되지 않습니다."
        )
    if cfg["alerts"].get("dry_run", True):
        st.warning("dry-run 모드입니다. 실제 발송 없이 본문만 표시합니다.")

    st.text_input("제목", message.subject, disabled=True)
    st.text_area("본문", message.body, height=320, disabled=True)

    if st.button("알림 발송", type="primary"):
        from yeda.alerts.email import send

        sent = send(message, cfg)
        if sent.sent:
            st.success(f"발송 완료: {', '.join(sent.recipients)}")
        elif sent.error:
            st.error(sent.error)
        else:
            st.info("dry-run 모드이므로 실제로 발송하지 않았습니다.")


def main() -> None:
    """앱 진입점."""
    cfg = _config()
    backend = _backend()

    st.title(cfg["app"]["title"])
    st.caption(cfg["app"]["subtitle"])

    if backend.is_mock:
        st.error(MOCK_BADGE)
    st.caption(
        "본 시스템은 실공정 데이터 접근이 불가하여 **공개 문헌 기반 합성 데이터**로 학습되었습니다. "
        "자세한 내용은 README 의 '데이터 출처' 절을 참조하세요."
    )

    values = _sidebar_inputs(cfg)
    tabs = st.tabs(["① 수율 예측", "② 원인 분해", "③ 개선 가이드", "④ 알림"])

    probability, risky, guides = 0.0, [], []

    with tabs[0]:
        try:
            probability = _render_predict(backend, values, cfg)
        except Exception as exc:  # noqa: BLE001 - 탭 하나가 죽어도 앱은 살아 있어야 한다
            st.exception(exc)

    with tabs[1]:
        try:
            risky = _render_explain(backend, values)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    with tabs[2]:
        try:
            guides = _render_optimize(backend, values)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    with tabs[3]:
        try:
            _render_alert(values, probability, risky, guides, cfg)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    st.sidebar.divider()
    st.sidebar.caption(backend.status)


if __name__ == "__main__":
    main()
