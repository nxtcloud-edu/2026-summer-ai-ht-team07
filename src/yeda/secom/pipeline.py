"""UCI SECOM 일반화 검증 — 실측 팹 데이터에서 파이프라인이 작동하는지 확인.

**범위 제한 (반드시 지킬 것)**
    SECOM 은 특성 590개가 전부 익명 센서값이라 **물리적 의미가 없다.**
    따라서 이 데이터로는 "무엇을 얼마로 바꾸라"는 최적화 가이드를 만들 수 없다.
    이 모듈의 포지션은 오직 하나다 — **"전처리→학습→평가 파이프라인이 실측
    팹 데이터에서도 돌아간다"는 일반화 검증.** 발표에서는 보조 슬라이드 1장 분량이다.

    이 선을 넘어 "SECOM 에서도 최적화했다"고 말하는 순간, 익명 변수에 대해 물리적
    조언을 했다는 지적을 받는다. 그 지적에는 방어할 방법이 없다.

**데이터 특성**
    1,567행 x 590특성, 결측 다수, 불량률 약 6.6% (극심한 불균형).
    → accuracy 는 무의미하다(전부 정상이라 찍어도 93.4%). **PR-AUC / recall 로만 본다.**

**메인 데모와의 분리**
    이 모듈은 ``yeda.schema`` 를 import 하지 않는다. SECOM 은 스키마가 완전히 다르며,
    억지로 공용 스키마에 끼워 맞추면 메인 데모가 망가진다.
    실행도 별도 명령(``make secom``)으로 분리한다.

소유자: B(계측·IoT) 주담당, C(데이터·모델) 백업.
우선순위: **낮음.** M4 까지 메인 데모가 완성되지 않았다면 이 모듈은 버린다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io_utils import EXTERNAL_DIR, resolve, save_json

SECOM_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
SECOM_LABEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

SCOPE_NOTE: str = (
    "SECOM 은 익명 센서 590개로 구성되어 변수의 물리적 의미를 알 수 없습니다. "
    "따라서 조건 최적화 가이드는 적용하지 않으며, 본 결과는 전처리·학습·평가 "
    "파이프라인이 실측 팹 데이터에서도 동작함을 보이는 일반화 검증으로만 사용합니다."
)
"""README·발표자료·UI에 그대로 넣을 범위 제한 문구."""


@dataclass
class SecomResult:
    """SECOM 검증 결과.

    Attributes:
        n_rows: 행 수.
        n_features: 특성 수.
        defect_rate: 불량률.
        metrics: PR-AUC / recall 중심 지표.
        baseline_pr_auc: 무작위 분류기의 PR-AUC(=불량률). 개선 폭 판단 기준선.
    """

    n_rows: int
    n_features: int
    defect_rate: float
    metrics: dict[str, float]
    baseline_pr_auc: float


def download(force: bool = False) -> tuple[Path, Path]:
    """SECOM 원본을 ``data/external/`` 로 내려받는다.

    Args:
        force: True 면 이미 있어도 다시 받는다.

    Returns:
        ``(데이터 경로, 라벨 경로)``.

    Raises:
        RuntimeError: 네트워크 실패 시. 해커톤장 네트워크를 신뢰할 수 없으므로,
            실패하면 이 파트를 통째로 건너뛰고 메인 데모에 집중한다.

    TODO(B): urllib 로 두 파일 저장. 실패 시 명확한 안내 메시지와 함께 RuntimeError.
    """
    raise NotImplementedError("TODO(B): SECOM 다운로드 구현 — scripts/download_secom.py 참고")


def load(data_path: Path | None = None, label_path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """SECOM 을 DataFrame 으로 읽는다.

    원본은 공백 구분에 결측이 ``NaN`` 문자열로 들어 있고, 라벨은 ``-1``(정상) /
    ``1``(불량) + 타임스탬프 형식이다. 라벨을 0/1 로 뒤집어 맞춘다.

    Args:
        data_path: ``secom.data`` 경로. None 이면 ``data/external/secom.data``.
        label_path: ``secom_labels.data`` 경로.

    Returns:
        ``(X, y)``. ``y`` 는 1 = 불량(양성 클래스).

    TODO(B): pd.read_csv(sep=r"\\s+", header=None) 로 로드. 라벨은 첫 컬럼만 사용.
    """
    raise NotImplementedError("TODO(B): SECOM 로드 구현")


def preprocess(X: pd.DataFrame, missing_threshold: float = 0.4) -> pd.DataFrame:
    """결측 과다 컬럼 제거 → 상수 컬럼 제거 → 중앙값 대치.

    590개 중 상당수가 결측률이 높거나 분산이 0이다. 이 정리를 안 하면 학습이
    느려지고 지표가 불안정해진다.

    Args:
        X: 원시 특성 DataFrame.
        missing_threshold: 이 비율 이상 결측인 컬럼은 버린다.

    Returns:
        정리된 DataFrame.

    TODO(B): 위 3단계 구현. 대치값은 학습 세트에서만 계산해 누수를 피할 것.
    """
    raise NotImplementedError("TODO(B): SECOM 전처리 구현")


def run(config: dict[str, Any] | None = None) -> SecomResult:
    """SECOM 전체 파이프라인을 돌리고 결과를 저장한다.

    ``make secom`` 진입점.

    Args:
        config: 선택적 설정. 기본값은 LightGBM + ``scale_pos_weight`` 로 불균형 보정.

    Returns:
        SecomResult.

    Note:
        메인 데모의 ``configs/model.yaml`` 을 재사용하지 않는다. 클래스 불균형이
        전혀 달라 하이퍼파라미터가 공유될 이유가 없고, 공유하면 메인 쪽을 건드리게 된다.

    TODO(B): load → preprocess → StratifiedKFold → LightGBM → PR-AUC/recall 보고.
             결과는 artifacts/metrics/secom_validation.json 으로 저장.
    """
    raise NotImplementedError("TODO(B): SECOM 파이프라인 구현")


def summary_markdown(result: SecomResult) -> str:
    """발표 슬라이드에 붙일 한 문단 요약을 만든다.

    Args:
        result: ``run()`` 결과.

    Returns:
        범위 제한 문구가 반드시 포함된 마크다운 문자열.
    """
    return (
        f"**SECOM 일반화 검증** — {result.n_rows:,}행 x {result.n_features}특성, "
        f"불량률 {result.defect_rate:.1%}. "
        f"PR-AUC {result.metrics.get('pr_auc', float('nan')):.3f} "
        f"(무작위 기준선 {result.baseline_pr_auc:.3f}), "
        f"recall {result.metrics.get('recall', float('nan')):.3f}.\n\n> {SCOPE_NOTE}"
    )
