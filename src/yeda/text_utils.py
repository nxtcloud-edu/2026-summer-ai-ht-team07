"""한국어 문장 조립 유틸리티.

최적화 가이드와 알림 메일이 모두 한국어 문장을 만든다. "핀 압력을(를)" 같은
표기는 발표 화면에서 완성도가 떨어져 보이므로 받침을 보고 조사를 고른다.

소유자: E(UI·통합).
"""

from __future__ import annotations

# 한글 음절 영역: 가(0xAC00) ~ 힣(0xD7A3). 종성 인덱스가 0이면 받침이 없다.
_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3
_JONGSUNG_COUNT = 28


def has_final_consonant(word: str) -> bool:
    """단어의 마지막 글자에 받침이 있는지 판정한다.

    Args:
        word: 대상 단어. 뒤쪽 공백·괄호는 무시한다.

    Returns:
        받침이 있으면 True. 한글이 아닌 문자로 끝나면 False (조사 기본형 사용).

    Note:
        숫자나 영문으로 끝나는 경우(예: "29N")는 판정이 불가능하므로 False 로 둔다.
        조사가 필요한 자리에는 되도록 한글 라벨을 넘길 것.
    """
    stripped = word.rstrip(" )]}")
    if not stripped:
        return False
    code = ord(stripped[-1])
    if not (_HANGUL_START <= code <= _HANGUL_END):
        return False
    return (code - _HANGUL_START) % _JONGSUNG_COUNT != 0


def josa(word: str, with_final: str, without_final: str) -> str:
    """받침 유무에 맞는 조사를 붙여 반환한다.

    Args:
        word: 앞 단어.
        with_final: 받침이 있을 때 쓸 조사 (예: ``"을"``).
        without_final: 받침이 없을 때 쓸 조사 (예: ``"를"``).

    Returns:
        ``단어 + 조사`` 문자열.

    Example:
        >>> josa("핀 압력", "을", "를")
        '핀 압력을'
        >>> josa("UV 조사 시간", "을", "를")
        'UV 조사 시간을'
    """
    return word + (with_final if has_final_consonant(word) else without_final)


def eul_reul(word: str) -> str:
    """목적격 조사 을/를 을 붙인다."""
    return josa(word, "을", "를")


def i_ga(word: str) -> str:
    """주격 조사 이/가 를 붙인다."""
    return josa(word, "이", "가")


def eun_neun(word: str) -> str:
    """보조사 은/는 을 붙인다."""
    return josa(word, "은", "는")


def format_value(value: float, unit: str) -> str:
    """값과 단위를 붙여 표시한다. 무단위(``-``)면 단위를 생략한다.

    Args:
        value: 표시할 수치.
        unit: ``schema.FeatureSpec.unit``.

    Returns:
        ``"29.5N"`` / ``"1"`` 같은 문자열.
    """
    text = f"{value:g}"
    return text if unit in ("-", "") else f"{text}{unit}"
