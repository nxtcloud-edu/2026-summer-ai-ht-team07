"""임계값 초과 시 이메일 알림.

.. important::
   **기본값은 ``dry_run: true``.** 실제 발송 없이 메일 본문만 만들어 반환한다.
   리허설을 반복하다 수신함을 채우거나, 발표 중 SMTP 접속 지연으로 화면이 멈추는
   사고를 막기 위함이다. 본 시연 직전에만 ``configs/app.yaml`` 에서 끈다.

.. warning::
   SMTP 자격증명은 ``.env`` 에서만 읽는다. YAML 이나 코드에 적지 않는다.
   리포지토리에 한 번 올라간 비밀번호는 되돌릴 수 없다.

소유자: E(UI·통합). 알림 문구 검수는 B(계측·IoT)가 맡는다.
"""

from __future__ import annotations

import os
import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any

from ..io_utils import load_config
from ..schema import SPEC_BY_NAME
from ..text_utils import format_value


@dataclass
class AlertMessage:
    """생성된 알림 메시지.

    Attributes:
        subject: 메일 제목.
        body: 메일 본문 (평문).
        recipients: 수신자 목록.
        sent: 실제로 발송되었는지 여부. ``dry_run`` 이면 항상 False.
        error: 발송 실패 사유. 성공이면 None.
    """

    subject: str
    body: str
    recipients: list[str] = field(default_factory=list)
    sent: bool = False
    error: str | None = None


# 같은 다이에 대한 반복 발송을 막는 메모리 캐시. {die_id: 마지막 발송 시각}
_last_sent: dict[str, float] = {}


def risk_level(probability: float, config: dict[str, Any] | None = None) -> str:
    """예측 성공확률을 위험 등급으로 바꾼다.

    Args:
        probability: 픽업 성공 확률 (0~1).
        config: ``configs/app.yaml``. None 이면 파일에서 읽는다.

    Returns:
        ``"critical"`` / ``"warning"`` / ``"normal"`` 중 하나 (``schema.RISK_LEVELS``).
    """
    cfg = config if config is not None else load_config("app")
    thresholds = cfg["risk_thresholds"]
    if probability < float(thresholds["critical"]):
        return "critical"
    if probability < float(thresholds["warning"]):
        return "warning"
    return "normal"


def should_alert(die_id: str, level: str, config: dict[str, Any] | None = None) -> bool:
    """알림을 보낼 조건인지 판정한다 (등급 + 쿨다운).

    Args:
        die_id: 대상 다이 식별자.
        level: ``risk_level()`` 결과.
        config: 앱 설정.

    Returns:
        보내야 하면 True.

    Note:
        쿨다운이 없으면 배치 100건을 돌릴 때 메일이 100통 나간다. 실무자가
        가장 빨리 알림을 꺼버리게 만드는 실수라 처음부터 넣어 둔다.
    """
    cfg = config if config is not None else load_config("app")
    alerts = cfg["alerts"]
    if not alerts.get("enabled", False):
        return False

    order = {"critical": 0, "warning": 1, "normal": 2}
    if order.get(level, 2) > order.get(alerts.get("trigger_level", "critical"), 0):
        return False

    cooldown = float(alerts.get("cooldown_seconds", 0))
    last = _last_sent.get(die_id)
    if last is not None and (time.time() - last) < cooldown:
        return False
    return True


def build_message(
    die_id: str,
    probability: float,
    risk_features: list[dict[str, Any]] | None = None,
    recommendations: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> AlertMessage:
    """알림 메일 본문을 만든다.

    본문에는 세 가지가 들어간다: 무엇이 문제인지(확률), 왜 그런지(SHAP 상위 변수),
    무엇을 하면 되는지(개선 가이드). 셋 중 하나라도 빠지면 실무자가 행동할 수 없다.

    Args:
        die_id: 다이 식별자.
        probability: 예측 픽업 성공 확률 (0~1).
        risk_features: ``explain.top_risk_features()`` 결과.
        recommendations: ``optimize.format_recommendations()`` 결과.
        config: 앱 설정.

    Returns:
        AlertMessage (아직 발송 전).
    """
    cfg = config if config is not None else load_config("app")
    alerts = cfg["alerts"]

    subject = alerts["subject_template"].format(die_id=die_id, probability=probability * 100)

    lines = [
        f"다이 {die_id} 의 예측 픽업 성공률이 {probability * 100:.1f}% 로 임계값을 밑돌았습니다.",
        "",
    ]

    if risk_features:
        lines.append("[주요 위험 변수 — SHAP 기여도]")
        for item in risk_features:
            spec = SPEC_BY_NAME.get(item["feature"])
            unit = spec.unit if spec else ""
            value = format_value(item["feature_value"], unit)
            lines.append(
                f"  - {item['korean']}: 현재 {value} → 성공률 {abs(item['shap_value_pp']):.1f}%p 감소 기여"
            )
        lines.append("")

    if recommendations:
        lines.append("[개선 가이드]")
        for text in recommendations:
            lines.append(f"  - {text}")
        lines.append("")

    lines += [
        "-" * 60,
        "※ 본 알림의 기여도 분석은 학습된 모델에 대한 상관 기반 설명(SHAP)이며 인과 효과 추정이 아닙니다.",
        "※ 개선 가이드는 조정 후보를 좁히기 위한 참고값입니다. 실제 적용 전 현장 검증이 필요합니다.",
        "※ 본 시스템은 공개 문헌 기반 합성 데이터로 학습되었습니다.",
        f"— {alerts.get('sender_name', 'YEDA')}",
    ]

    return AlertMessage(subject=subject, body="\n".join(lines), recipients=_resolve_recipients(cfg))


def _resolve_recipients(config: dict[str, Any]) -> list[str]:
    """수신자 목록을 결정한다 (.env 우선, 없으면 설정 기본값)."""
    env_value = os.environ.get("ALERT_RECIPIENTS", "").strip()
    if env_value:
        return [addr.strip() for addr in env_value.split(",") if addr.strip()]
    return list(config["alerts"].get("default_recipients") or [])


def send(message: AlertMessage, config: dict[str, Any] | None = None) -> AlertMessage:
    """메시지를 발송한다. ``dry_run`` 이면 발송하지 않고 그대로 돌려준다.

    Args:
        message: ``build_message()`` 결과.
        config: 앱 설정.

    Returns:
        ``sent`` / ``error`` 가 채워진 같은 객체.

    Note:
        예외를 밖으로 던지지 않는다. 알림 실패로 데모가 멈추면 안 되기 때문에
        실패 사유는 ``error`` 에 담아 UI 가 조용히 표시하게 한다.
    """
    cfg = config if config is not None else load_config("app")
    alerts = cfg["alerts"]

    if alerts.get("dry_run", True):
        message.error = None
        message.sent = False
        return message

    if not message.recipients:
        message.error = "수신자가 없습니다 (.env 의 ALERT_RECIPIENTS 를 설정하세요)"
        return message

    smtp_cfg = alerts["smtp"]
    host = os.environ.get(smtp_cfg["host_env"], "")
    port = int(os.environ.get(smtp_cfg["port_env"], "587") or 587)
    user = os.environ.get(smtp_cfg["user_env"], "")
    password = os.environ.get(smtp_cfg["password_env"], "")

    if not host or not user or not password:
        message.error = "SMTP 자격증명이 없습니다 (.env 확인). dry_run 모드로 두세요."
        return message

    email = EmailMessage()
    email["Subject"] = message.subject
    email["From"] = f"{alerts.get('sender_name', 'YEDA')} <{user}>"
    email["To"] = ", ".join(message.recipients)
    email.set_content(message.body)

    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            if smtp_cfg.get("use_tls", True):
                server.starttls()
            server.login(user, password)
            server.send_message(email)
        message.sent = True
    except Exception as exc:  # noqa: BLE001 - 알림 실패가 데모를 멈추면 안 된다
        message.error = f"발송 실패: {type(exc).__name__}: {exc}"
    return message


def notify(
    die_id: str,
    probability: float,
    risk_features: list[dict[str, Any]] | None = None,
    recommendations: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> AlertMessage | None:
    """판정 → 본문 생성 → 발송까지 한 번에 처리한다 (UI 진입점).

    Args:
        die_id: 다이 식별자.
        probability: 예측 성공 확률.
        risk_features: SHAP 상위 위험 변수.
        recommendations: 개선 가이드 문장.
        config: 앱 설정.

    Returns:
        발송(또는 dry-run) 결과. 알림 조건이 아니면 None.
    """
    cfg = config if config is not None else load_config("app")
    level = risk_level(probability, cfg)
    if not should_alert(die_id, level, cfg):
        return None

    message = send(build_message(die_id, probability, risk_features, recommendations, cfg), cfg)
    _last_sent[die_id] = time.time()
    return message
