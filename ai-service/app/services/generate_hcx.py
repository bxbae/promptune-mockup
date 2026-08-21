from __future__ import annotations

import logging
import re

import torch

from app.schemas.models import GenerateRequest, GenerateResponse
from app.services.hcx_runtime import HCX_MODEL_LOCK, load_hcx_runtime
from app.services.retrieval.retrieval_context import build_internal_context


logger = logging.getLogger(__name__)


def _build_internal_context(
    req: GenerateRequest,
) -> str:
    return build_internal_context(req.documents)


def _build_web_context(web_results: list[dict]) -> str:
    if not web_results:
        return "없음"

    parts: list[str] = []

    for index, item in enumerate(web_results, start=1):
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()

        if len(content) > 1200:
            content = content[:1200]

        parts.append(
            f"[웹 검색 결과 {index}]\n"
            f"제목: {title}\n"
            f"URL: {url}\n"
            f"내용: {content}"
        )
    return "\n\n".join(parts)


def _build_user_context(user_context: dict[str, str]) -> str:
    if not user_context:
        return "없음"

    labels = {
        "displayName": "이름",
        "companyName": "회사",
        "department": "부서",
        "jobTitle": "직급/직책",
        "mail": "이메일",
    }

    parts = []
    for key, value in user_context.items():
        value = str(value or "").strip()
        if value:
            parts.append(f"{labels.get(key, key)}: {value}")

    return "\n".join(parts) if parts else "없음"



def _build_preference_context(preference: dict[str, str]) -> str:
    if not preference:
        return "없음"

    labels = {
        "speed": "속도",
        "detail": "설명 분량",
        "preserve": "원문 존중도",
    }

    value_labels = {
        "fast": "빠르게",
        "accurate": "정확하게",
        "brief": "간결하게",
        "detailed": "자세하게",
        "keep": "원문 최대한 유지",
        "improve": "적극적으로 보완",
    }

    parts: list[str] = []

    for key, value in preference.items():
        value = str(value or "").strip()

        if value:
            label = labels.get(key, key)
            value_label = value_labels.get(value, value)
            parts.append(f"{label}: {value_label}")

    return "\n".join(parts) if parts else "없음"



def _build_effective_user_prompt(req: GenerateRequest) -> str:
    """
    Retrieval에서는 원본 사용자 요청을 사용하고,
    Generation에서는 이미 검색된 문서의 파일명(locator)을 제거해
    실제 수행할 요청만 HCX에 전달한다.
    """
    original = req.prompt.strip()

    if not original or not req.documents:
        return original

    titles = sorted(
        {
            document.title.strip()
            for document in req.documents
            if document.title and document.title.strip()
        },
        key=len,
        reverse=True,
    )

    normalized = original
    matched_title = False

    for title in titles:
        candidates = [
            title + "의 ",
            title + "에서 ",
            title + "을 ",
            title + "를 ",
            title + "으로 ",
            title + "로 ",
            title,
        ]

        for candidate in candidates:
            if candidate in normalized:
                normalized = normalized.replace(candidate, "", 1)
                matched_title = True
                break

    if not matched_title:
        return original

    prefixes = [
        "내부 문서에서 ",
        "내부문서에서 ",
        "사내 문서에서 ",
        "사내문서에서 ",
        "회사 문서에서 ",
        "회사문서에서 ",
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break

    normalized = " ".join(normalized.split()).strip()

    if not normalized:
        normalized = "핵심 내용을 알려줘"

    return "제공된 내부 문서에서 " + normalized


def _build_prompt(
    req: GenerateRequest,
    web_results: list[dict],
) -> str:
    internal_context = _build_internal_context(req)
    web_context = _build_web_context(web_results)
    user_context = _build_user_context(req.user_context)
    preference_context = _build_preference_context(req.preference)
    user_prompt = _build_effective_user_prompt(req)

    parts = [
        "너는 업무용 AI 어시스턴트다.",
        "사용자의 요청과 실제로 제공된 참고자료를 바탕으로 직접 답변해.",
        "",
        "규칙:",
        "1. 참고자료에 없는 사실을 임의로 만들어내지 마.",
        "2. 내부 문서가 제공되면 내부 문서의 관련 내용을 최우선 근거로 사용해.",
        "3. 내부 문서가 제공된 경우 '문서를 확인할 수 없다'거나 '자료가 없다'고 답하지 마.",
        "4. 사용자가 내부 문서의 핵심 내용이나 요약을 요청하면 내부 문서의 실제 내용을 구체적으로 요약해.",
        "5. 웹 검색 결과는 실제로 제공된 경우에만 활용해.",
        "6. 웹 검색 결과가 제공되지 않은 경우 그 사실을 답변에서 언급하지 마.",
        "7. 사용자 프로필은 실제로 제공된 경우에만 활용해.",
        "8. 최종 답변만 출력하고 분석 과정은 출력하지 마.",
        "9. 사용자 선호도가 제공된 경우, 그 스타일(속도/설명 분량/원문 존중도)에 맞춰 답변을 조정해.",
        "",
        f"[업무 유형]\n{req.task_type}",
        "",
        f"[사용자 요청]\n{user_prompt}",
    ]

    if internal_context != "없음":
        parts.extend([
            "",
            "[내부 문서 - 아래 내용을 반드시 답변 근거로 활용]",
            internal_context,
        ])

    if web_context != "없음":
        parts.extend([
            "",
            "[웹 검색 결과]",
            web_context,
        ])

    if user_context != "없음":
        parts.extend([
            "",
            "[사용자 프로필]",
            user_context,
        ])

    if preference_context != "없음":
        parts.extend([
            "",
            "[사용자 선호도]",
            preference_context,
        ])

    parts.extend([
        "",
        "[최종 답변]",
    ])

    return "\n".join(parts)

def generate(
    req: GenerateRequest,
    web_results=None,
    used_web_search: bool = False,
) -> GenerateResponse:
    web_results = web_results or []

    tokenizer, model, device = load_hcx_runtime()

    prompt = _build_prompt(req=req, web_results=web_results)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    with HCX_MODEL_LOCK:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                stop_strings=[
                    "<|endofturn|>",
                    "<|stop|>",
                ],
                tokenizer=tokenizer,
            )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    result = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    logger.info(
        "HCX final generation task_type=%s web=%s documents=%d",
        req.task_type,
        used_web_search,
        len(req.documents),
    )

    return GenerateResponse(
        result=result,
        used_web_search=used_web_search,
    )
