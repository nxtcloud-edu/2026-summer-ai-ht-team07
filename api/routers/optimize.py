"""최적화 라우터.

엔드포인트:
    POST /api/optimize — 현재 조건 → 개선 가이드

소유자: D(API 라우터·설명·최적화).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.models.request import OptimizeRequest
from api.models.response import OptimizeResponse, RecommendationItem
from api.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/api", tags=["optimize"])


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest, service: MLService = Depends(get_ml_service)):
    """현재 공정 조건에 대한 개선 가이드를 생성한다."""
    try:
        values = request.to_feature_dict()
        result = service.optimize(values)
        recommendations = [RecommendationItem(**item) for item in result["recommendations"]]
        return OptimizeResponse(
            baseline_prob=result["baseline_prob"],
            optimized_prob=result["optimized_prob"],
            gain_pp=result["gain_pp"],
            recommendations=recommendations,
            formatted_text=result["formatted_text"],
            n_evaluations=result["n_evaluations"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"피처 누락: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"최적화 실패: {exc}") from exc
