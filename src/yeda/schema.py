"""YEDA 공용 데이터 계약(contract).

이 파일은 **모든 모듈이 공유하는 단 하나의 진실 원천(single source of truth)** 이다.
컬럼명, 물리적 허용 범위, 센서 분해능, 조정 가능 여부, 단조성 방향이 여기에 모여 있다.

.. warning::
   이 파일은 킥오프 1시간 안에 확정하고 **그 이후에는 변경을 최소화한다.**
   변경이 필요하면 반드시 팀 전체에 공지하고 통합 담당자(E)가 머지한다.
   (5명 동시 작업 중 git 충돌 위험이 가장 높은 파일)

소유자: E(UI·통합) — 단, 물리 값(RANGE / MONOTONE / RESOLUTION)의 내용 결정권은
        A(공정물리) + B(계측·IoT)에게 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------
# 1. 타겟 / 식별자
# --------------------------------------------------------------------------

TARGET: str = "pickup_success"
"""타겟 컬럼명. 1 = 픽업 성공, 0 = 픽업 실패."""

ID_COL: str = "die_id"
"""개별 다이 식별자. 생성기가 부여하며 학습 피처에서는 제외된다."""


# --------------------------------------------------------------------------
# 2. 변수 스펙
# --------------------------------------------------------------------------

Kind = Literal["continuous", "categorical"]
"""변수 자료형 구분."""

Monotone = Literal[-1, 0, 1]
"""LightGBM ``monotone_constraints`` 방향.

+1 : 값이 커질수록 성공 확률 증가 (단조 증가)
-1 : 값이 커질수록 성공 확률 감소 (단조 감소)
 0 : 제약 없음 (내부 최적점이 존재하거나 물리적 방향이 확정되지 않음)

.. note::
   ``head_vacuum`` / ``pin_vacuum`` / ``vacuum_status`` 는 **부호가 음수인 kPa 게이지압**이다.
   "진공이 세다" = "값이 더 작다(더 음수)" 이므로, 진공이 셀수록 성공률이 오르는
   물리 관계는 **부호 있는 값 기준으로는 단조 감소(-1)** 로 표현된다.
   이 부호 규약을 놓치면 제약 방향이 통째로 뒤집힌다. 반드시 확인할 것.
"""


@dataclass(frozen=True)
class FeatureSpec:
    """공정 변수 하나에 대한 명세.

    Attributes:
        name: 컬럼명.
        unit: 표시 단위 (UI 라벨에 그대로 사용).
        low: 물리적 허용 하한 (정상 운전 범위).
        high: 물리적 허용 상한.
        kind: 연속형 / 범주형.
        adjustable: 설비에서 실시간 조정 가능한가.
            False 인 변수(제품 사양·자재)는 **최적화 탐색 공간에서 제외**된다.
        monotone: 성공 확률에 대한 단조성 방향 (위 ``Monotone`` 참고).
        resolution: 센서/설정 분해능. 데이터 생성 시 이 단위로 반올림하고,
            최적화 제안값도 이 단위로 스냅한다. (0.01 → 소수 둘째 자리)
        step: UI 슬라이더 및 최적화 그리드 기본 간격.
        korean: UI 표기용 한글명.
        rationale: 물리적 근거 요약 한 줄. 발표 Q&A에서 그대로 인용 가능해야 한다.
    """

    name: str
    unit: str
    low: float
    high: float
    kind: Kind
    adjustable: bool
    monotone: Monotone
    resolution: float
    step: float
    korean: str
    rationale: str


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="uv_time",
        unit="s",
        low=3.0,
        high=7.0,
        kind="continuous",
        adjustable=True,
        monotone=1,
        resolution=0.1,
        step=0.1,
        korean="UV 조사 시간",
        rationale="UV 조사가 길수록 다이싱 테이프의 점착제가 경화되어 점착력이 낮아지고 "
        "이형이 쉬워진다. 다만 효과는 포화되며 과조사 시 잔사(residue) 위험이 있다.",
    ),
    FeatureSpec(
        name="uv_intensity",
        unit="mW/cm^2",
        low=1000.0,
        high=1400.0,
        kind="continuous",
        adjustable=True,
        monotone=1,
        resolution=1.0,
        step=10.0,
        korean="UV 광량",
        rationale="총 조사 에너지 = 광량 x 시간. uv_time 과 같은 경화 메커니즘이며 "
        "기여도는 더 작다(설비 램프 출력 편차 범위가 좁음).",
    ),
    FeatureSpec(
        name="pin_speed",
        unit="mm/s",
        low=0.8,
        high=1.6,
        kind="continuous",
        adjustable=True,
        monotone=-1,
        resolution=0.01,
        step=0.05,
        korean="이젝터 핀 상승 속도",
        rationale="핀이 빠를수록 다이에 걸리는 순간 굽힘 응력이 커져 크랙/치핑 위험이 증가한다. "
        "얇은 다이일수록 이 효과가 커진다(die_thickness 와 상호작용).",
    ),
    FeatureSpec(
        name="pin_pressure",
        unit="N",
        low=20.0,
        high=40.0,
        kind="continuous",
        adjustable=True,
        monotone=1,
        resolution=0.5,
        step=0.5,
        korean="핀 압력",
        rationale="테이프-다이 계면을 떼어내는 구동력. 클수록 이형 성공률이 오르나 포화된다. "
        "(크랙 위험은 속도/높이 쪽으로 분리해 모델링)",
    ),
    FeatureSpec(
        name="pin_height",
        unit="mm",
        low=0.3,
        high=0.7,
        kind="continuous",
        adjustable=True,
        monotone=0,
        resolution=0.01,
        step=0.01,
        korean="핀 상승 높이",
        rationale="너무 낮으면 계면 박리가 시작되지 않고, 너무 높으면 다이가 과굽힘되어 파손된다. "
        "→ 내부 최적점 존재 → 단조 제약 없음(0).",
    ),
    FeatureSpec(
        name="head_vacuum",
        unit="kPa",
        low=-70.0,
        high=-60.0,
        kind="continuous",
        adjustable=True,
        monotone=-1,
        resolution=0.1,
        step=0.5,
        korean="픽업 헤드 진공압",
        rationale="콜릿이 다이를 흡착 유지하는 힘. 값이 더 음수일수록 흡착이 강해 "
        "이형 직후 다이를 놓치지 않는다. (부호 규약상 단조 감소)",
    ),
    FeatureSpec(
        name="pin_vacuum",
        unit="kPa",
        low=-55.0,
        high=-45.0,
        kind="continuous",
        adjustable=True,
        monotone=-1,
        resolution=0.1,
        step=0.5,
        korean="이젝터 진공압",
        rationale="이젝터 테이블이 테이프를 하부에서 고정하는 힘. 강할수록 테이프가 들뜨지 않아 "
        "핀 상승 시 박리가 국부적으로 집중된다. (부호 규약상 단조 감소)",
    ),
    FeatureSpec(
        name="temperature",
        unit="degC",
        low=20.0,
        high=30.0,
        kind="continuous",
        adjustable=True,
        monotone=0,
        resolution=0.1,
        step=0.5,
        korean="공정 온도",
        rationale="점착제는 온도 의존 점탄성을 가진다. 너무 낮으면 취성 증가, 너무 높으면 "
        "점착력이 다시 올라가 이형이 어려워진다 → 내부 최적점 존재.",
    ),
    FeatureSpec(
        name="humidity",
        unit="%RH",
        low=30.0,
        high=60.0,
        kind="continuous",
        adjustable=True,
        monotone=-1,
        resolution=1.0,
        step=1.0,
        korean="상대습도",
        rationale="습도가 높을수록 계면 수분이 점착 특성을 바꾸고 미세 슬립을 유발한다. "
        "(저습 측 정전기 이슈는 이번 범위에서 제외 — 한계로 문서화)",
    ),
    FeatureSpec(
        name="vacuum_status",
        unit="kPa",
        low=-100.0,
        high=-90.0,
        kind="continuous",
        adjustable=False,
        monotone=-1,
        resolution=0.1,
        step=0.5,
        korean="진공 시스템 상태",
        rationale="설비 공용 진공 라인의 건전성 지표. 개별 레시피로 조정하는 값이 아니라 "
        "설비 상태를 나타내므로 **조정 불가**로 둔다(설비 보전 대상).",
    ),
    FeatureSpec(
        name="runtime_hours",
        unit="h",
        low=0.0,
        high=24.0,
        kind="continuous",
        adjustable=False,
        monotone=-1,
        resolution=0.1,
        step=0.5,
        korean="설비 연속 가동 시간",
        rationale="콜릿 고무 마모/오염의 대리 지표. 길수록 흡착면 밀착이 나빠져 성공률이 떨어진다. "
        "레시피로 못 바꾸고 콜릿 교체로만 리셋되므로 **조정 불가**.",
    ),
    FeatureSpec(
        name="tape_type",
        unit="-",
        low=0.0,
        high=1.0,
        kind="categorical",
        adjustable=False,
        monotone=0,
        resolution=1.0,
        step=1.0,
        korean="다이싱 테이프 종류",
        rationale="0 = 일반 점착 테이프, 1 = UV 경화형 테이프. 자재 사양이므로 조정 불가. "
        "UV 반응 민감도가 달라 uv_time 과 상호작용한다.",
    ),
    FeatureSpec(
        name="die_thickness",
        unit="um",
        low=100.0,
        high=150.0,
        kind="continuous",
        adjustable=False,
        monotone=1,
        resolution=1.0,
        step=1.0,
        korean="다이 두께",
        rationale="두꺼울수록 굽힘 강성이 커 크랙에 강하다. **제품 사양이므로 조정 불가** — "
        "이 변수를 '바꾸라'고 제안하는 순간 실무자에게 신뢰를 잃는다.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
"""학습에 사용하는 피처 순서. **모든 모듈이 이 순서를 따른다.**

sklearn/LightGBM 은 컬럼 순서에 의존하므로, 저장된 모델과 추론 입력의 순서가
어긋나면 조용히 틀린 예측이 나온다. DataFrame 은 항상 ``df[list(FEATURE_NAMES)]`` 로 정렬할 것.
"""

SPEC_BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}

ADJUSTABLE: tuple[str, ...] = tuple(f.name for f in FEATURES if f.adjustable)
"""최적화 탐색 공간에 들어가는 변수(설비 파라미터)."""

FIXED: tuple[str, ...] = tuple(f.name for f in FEATURES if not f.adjustable)
"""고정 변수(제품 사양·자재·설비 상태). 최적화가 절대 건드리면 안 된다."""

CATEGORICAL: tuple[str, ...] = tuple(f.name for f in FEATURES if f.kind == "categorical")

BOUNDS: dict[str, tuple[float, float]] = {f.name: (f.low, f.high) for f in FEATURES}
"""물리적 허용 범위. 최적화 제약과 입력 검증 양쪽에서 사용."""

MONOTONE_CONSTRAINTS: list[int] = [f.monotone for f in FEATURES]
"""``FEATURE_NAMES`` 와 같은 순서의 LightGBM monotone_constraints 리스트.

이 리스트는 **도메인 지식에서만 유도**되었다. 데이터 생성기의 계수를 참조하지 않는다.
(생성기가 없어도 현직자 면담만으로 동일하게 채울 수 있어야 정당한 제약이다.)
"""

RESOLUTION: dict[str, float] = {f.name: f.resolution for f in FEATURES}


# --------------------------------------------------------------------------
# 3. 모듈 간 교환 스키마 (DataFrame 계약)
# --------------------------------------------------------------------------

RAW_COLUMNS: tuple[str, ...] = (ID_COL, *FEATURE_NAMES, TARGET)
"""``data/raw/yeda_synthetic.csv`` 의 컬럼. 생성기 출력 계약."""

PREDICTION_COLUMNS: tuple[str, ...] = (ID_COL, "y_prob", "y_pred", "risk_level")
"""``predict_frame()`` 출력 계약. ``y_prob`` 는 [0,1] 픽업 성공 확률."""

EXPLANATION_COLUMNS: tuple[str, ...] = (ID_COL, "feature", "shap_value_pp", "feature_value", "direction")
"""``explain_frame()`` 출력 계약(long format).

``shap_value_pp`` 는 **확률 %p 단위**다. 로그 오즈가 아니다.
(``shap.TreeExplainer(..., model_output="probability")`` 필수 — 4.3 참조)
"""

RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    "feature",
    "current_value",
    "suggested_value",
    "delta",
    "unit",
    "expected_gain_pp",
)
"""``recommend()`` 출력 계약. "무엇을 얼마로 바꾸면 몇 %p 오른다" 한 줄 = 한 행."""

RISK_LEVELS: tuple[str, ...] = ("critical", "warning", "normal")
"""``risk_level`` 허용값. 임계값은 ``configs/app.yaml`` 에서 관리한다."""


# --------------------------------------------------------------------------
# 4. 검증 헬퍼
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """스키마 검증 결과.

    Attributes:
        ok: 오류가 하나도 없으면 True.
        errors: 계약 위반(컬럼 누락, 타겟 값 이상 등). 반드시 고쳐야 함.
        warnings: 범위 이탈·결측 등 진행은 가능한 경고.
    """

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_frame(df, *, require_target: bool = True) -> ValidationReport:
    """DataFrame 이 YEDA 스키마 계약을 지키는지 검사한다.

    Args:
        df: 검사 대상 ``pandas.DataFrame``.
        require_target: True 면 ``pickup_success`` 컬럼 존재를 강제한다.
            추론 입력(라벨 없음)을 검사할 때는 False.

    Returns:
        ValidationReport: ``ok`` 가 False 면 ``errors`` 를 사용자에게 보여준다.

    Note:
        범위 이탈은 error 가 아니라 **warning** 이다. 실제 설비는 스펙 밖 값을
        찍을 수 있고, 데모 중 예외로 앱이 죽는 것이 더 나쁘기 때문이다.
    """
    report = ValidationReport()

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        report.add_error(f"필수 피처 누락: {missing}")

    if require_target and TARGET not in df.columns:
        report.add_error(f"타겟 컬럼 누락: {TARGET}")
    elif TARGET in df.columns:
        bad = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad:
            report.add_error(f"{TARGET} 에 0/1 이외 값 존재: {sorted(bad)}")

    for name in FEATURE_NAMES:
        if name not in df.columns:
            continue
        col = df[name]
        low, high = BOUNDS[name]
        out_of_range = int(((col < low) | (col > high)).sum())
        if out_of_range:
            report.add_warning(f"{name}: 허용 범위 [{low}, {high}] 이탈 {out_of_range} 행")
        na_ratio = float(col.isna().mean())
        if na_ratio > 0.05:
            report.add_warning(f"{name}: 결측률 {na_ratio:.1%} (>5%)")

    return report


def clip_to_bounds(values: dict[str, float]) -> dict[str, float]:
    """제안값을 물리적 허용 범위로 자르고 센서 분해능으로 스냅한다.

    최적화기가 "핀 압력 37.8213N" 같은 값을 뱉으면 실무자가 설정할 수 없다.
    제안은 반드시 설비가 실제로 받을 수 있는 값이어야 한다.

    Args:
        values: ``{피처명: 값}``.

    Returns:
        범위 클리핑 + 분해능 반올림이 적용된 새 dict. 알 수 없는 키는 그대로 통과.
    """
    out: dict[str, float] = {}
    for name, value in values.items():
        spec = SPEC_BY_NAME.get(name)
        if spec is None:
            out[name] = value
            continue
        clipped = min(max(float(value), spec.low), spec.high)
        snapped = round(clipped / spec.resolution) * spec.resolution
        # 부동소수 잔여 오차만 제거한다.
        #
        # 여기서 log10 으로 소수 자릿수를 추정하면 안 된다. 분해능이 10의 거듭제곱이
        # 아닌 경우(pin_pressure = 0.5N)에 자릿수가 0으로 계산되어, 애써 스냅한 33.5 가
        # 34.0 으로 다시 반올림된다. 그러면 분해능도 이동 폭 제한도 함께 깨진다.
        # (tests/test_optimize.py::test_move_limit_respected 가 이 회귀를 잡는다)
        out[name] = round(min(max(snapped, spec.low), spec.high), 10)
    return out
