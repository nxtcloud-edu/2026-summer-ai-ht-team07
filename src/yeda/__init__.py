"""Yield X — 다이 픽업 공정 수율 최적화 시스템.

모듈 구성:
    schema      공용 데이터 계약 (컬럼·범위·조정가능 여부·단조성). 모든 모듈의 기준.
    data        합성 데이터 생성 및 전처리
    models      학습 / 평가 / 모델 레지스트리
    explain     SHAP 기반 실패 원인 분해
    optimize    조정 가능 변수 탐색 및 개선 가이드
    alerts      임계값 초과 시 이메일 알림
    secom       UCI SECOM 일반화 검증 (메인 데모와 완전 분리)
"""

__version__ = "0.1.0"
