from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from services.config import GEMINI_API_KEY, GEMINI_API_TIMEOUT, GEMINI_MODEL


logger = logging.getLogger(__name__)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

REQUIRED_SCHEMA = {
    "match_score": 0,
    "matching_skills": [],
    "missing_skills": [],
    "experience_analysis": "",
    "education_analysis": "",
    "strengths": [],
    "weaknesses": [],
    "recommendation": "",
    "justification": "",
}


class LLMServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _build_prompt(resume_text: str, job_description: str) -> str:
    return f"""
You are an expert AI recruitment assistant.

Compare the following resume with the provided job description.

Analyze:

* Technical Skills
* Soft Skills
* Experience
* Education
* Certifications
* Projects
* Missing Skills
* Overall Suitability

Return ONLY valid JSON.

```json
{{
  "match_score": 0,
  "matching_skills": [],
  "missing_skills": [],
  "experience_analysis": "",
  "education_analysis": "",
  "strengths": [],
  "weaknesses": [],
  "recommendation": "",
  "justification": ""
}}
```

Resume:
{resume_text}

Job Description:
{job_description}
""".strip()


def _extract_response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise LLMServiceError("Gemini did not return any candidates.", status_code=502)

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        finish_reason = candidates[0].get("finishReason", "unknown")
        raise LLMServiceError(f"Gemini returned an empty response. Finish reason: {finish_reason}", status_code=502)
    return text


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise LLMServiceError("Gemini response did not contain valid JSON.", status_code=502)
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMServiceError("Gemini response JSON could not be parsed.", status_code=502) from exc


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _validate_result(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LLMServiceError("Gemini returned JSON that is not an object.", status_code=502)

    result = REQUIRED_SCHEMA.copy()
    result.update({key: data.get(key, default) for key, default in REQUIRED_SCHEMA.items()})

    try:
        result["match_score"] = int(round(float(result["match_score"])))
    except (TypeError, ValueError):
        raise LLMServiceError("Gemini returned an invalid match_score.", status_code=502)

    result["match_score"] = max(0, min(100, result["match_score"]))
    for key in ["matching_skills", "missing_skills", "strengths", "weaknesses"]:
        result[key] = _as_string_list(result.get(key))

    for key in ["experience_analysis", "education_analysis", "recommendation", "justification"]:
        result[key] = str(result.get(key) or "").strip()

    if not result["recommendation"]:
        result["recommendation"] = "Review"
    if not result["justification"]:
        result["justification"] = "Gemini returned a score but did not include a detailed justification."

    return result


def analyze_resume_match(resume_text: str, job_description: str) -> dict[str, Any]:
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()

    if not resume_text:
        raise LLMServiceError("Resume text is required for LLM matching.", status_code=400)
    if not job_description:
        raise LLMServiceError("Job description is required for LLM matching.", status_code=400)
    if not GEMINI_API_KEY:
        raise LLMServiceError("GEMINI_API_KEY is not configured.", status_code=503)

    url = GEMINI_ENDPOINT.format(model=GEMINI_MODEL)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _build_prompt(resume_text, job_description)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=GEMINI_API_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise LLMServiceError("Gemini request timed out.", status_code=504) from exc
    except requests.RequestException as exc:
        logger.exception("Gemini request failed")
        raise LLMServiceError("Gemini API request failed.", status_code=503) from exc

    if response.status_code == 429:
        raise LLMServiceError("Gemini rate limit exceeded. Try again later.", status_code=429)
    if response.status_code >= 400:
        logger.warning("Gemini API error %s: %s", response.status_code, response.text[:500])
        raise LLMServiceError("Gemini API returned an error.", status_code=502)

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise LLMServiceError("Gemini returned a non-JSON API response.", status_code=502) from exc

    raw_text = _extract_response_text(response_payload)
    return _validate_result(_parse_json(raw_text))
