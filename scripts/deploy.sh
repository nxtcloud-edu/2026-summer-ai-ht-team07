#!/bin/bash
# =============================================================================
# Yield X — 3-Tier 배포 스크립트
# =============================================================================
# 사용법:
#   1. 프로젝트 루트의 .env 에 배포 변수를 채운다
#   2. bash scripts/deploy.sh [frontend|backend|all]
#
# 필요한 .env 변수:
#   S3_BUCKET       — 프론트엔드 S3 버킷명 (예: yieldx-frontend-prod)
#   EC2_HOST        — 백엔드 EC2 퍼블릭 IP 또는 도메인
#   EC2_USER        — SSH 접속 유저 (기본: ec2-user)
#   EC2_KEY         — SSH 키 경로 (예: ~/.ssh/yieldx.pem)
#   DEPLOY_PATH     — EC2 내 앱 경로 (기본: /home/ec2-user/yieldx)
#
# 담당: B (DB · 인프라). 템플릿 작성: D.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# .env 로드 (루트)
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

# 기본값
EC2_USER="${EC2_USER:-ec2-user}"
DEPLOY_PATH="${DEPLOY_PATH:-/home/ec2-user/yieldx}"

# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------
check_vars() {
    local missing=()
    [ -z "${S3_BUCKET:-}" ] && missing+=("S3_BUCKET")
    [ -z "${EC2_HOST:-}" ] && missing+=("EC2_HOST")
    [ -z "${EC2_KEY:-}" ] && missing+=("EC2_KEY")

    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: 다음 환경변수가 .env에 없습니다: ${missing[*]}"
        echo "       프로젝트 루트의 .env.example 을 참고하세요."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Tier 1: Frontend → S3
# ---------------------------------------------------------------------------
deploy_frontend() {
    echo "=== [Tier 1] Frontend → S3 ($S3_BUCKET) ==="
    if [ ! -d "$ROOT_DIR/frontend" ]; then
        echo "WARN: frontend/ 디렉토리가 없습니다. E 파트 구현 후 재시도하세요."
        return 0
    fi
    aws s3 sync "$ROOT_DIR/frontend/" "s3://$S3_BUCKET" \
        --delete \
        --exclude ".*"
    echo "  ✓ S3 업로드 완료"
}

# ---------------------------------------------------------------------------
# Tier 2: Backend → EC2
# ---------------------------------------------------------------------------
deploy_backend() {
    echo "=== [Tier 2] Backend → EC2 ($EC2_HOST) ==="
    local SSH="ssh -i $EC2_KEY -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST"

    # 코드 동기화
    $SSH "cd $DEPLOY_PATH && git pull origin main"

    # 의존성 업데이트
    $SSH "cd $DEPLOY_PATH && source .venv/bin/activate && pip install -r requirements.txt -q"

    # API 서버 재시작 (systemd 서비스 가정)
    $SSH "sudo systemctl restart yieldx-api"

    echo "  ✓ Backend 배포 완료"
    echo "  ✓ API: http://$EC2_HOST:8000/api/health"
}

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
ACTION="${1:-all}"

check_vars

case "$ACTION" in
    frontend)
        deploy_frontend
        ;;
    backend)
        deploy_backend
        ;;
    all)
        deploy_frontend
        deploy_backend
        ;;
    *)
        echo "사용법: bash scripts/deploy.sh [frontend|backend|all]"
        exit 1
        ;;
esac

echo ""
echo "=== 배포 완료 ==="
