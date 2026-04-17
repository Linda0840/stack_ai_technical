"""
Query refusal policies.

Checks the raw user query for three categories of sensitive content and
returns a (category, user-facing message) pair when a match is found, or
(None, None) when the query is safe to process.

Design notes
------------
- Matching is case-insensitive and word-boundary-anchored so "address" does
  not accidentally match "addressed" or "legal" does not block "illegal" while
  still refusing the plain keyword.
- The check runs *before* any LLM or search call, so it adds zero latency to
  the happy path and produces no unnecessary token spend.
- Only the first matching category is returned (PII > legal > medical priority).
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── Keyword lists ─────────────────────────────────────────────────────────────

PII_KEYWORDS: list[str] = [
    "email",
    "phone",
    "ssn",
    "social security",
    "address",
    "contact",
    "passport",
    "id number",
]

LEGAL_KEYWORDS: list[str] = [
    "law",
    "legal",
    "contract",
    "lease",
    "liability",
]

MEDICAL_KEYWORDS: list[str] = [
    "symptoms",
    "diagnosis",
    "treatment",
    "medicine",
    "pain",
]

# ── Pre-compiled patterns ─────────────────────────────────────────────────────

def _build_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Word-boundary pattern that matches any keyword in the list."""
    alternation = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_PII_RE     = _build_pattern(PII_KEYWORDS)
_LEGAL_RE   = _build_pattern(LEGAL_KEYWORDS)
_MEDICAL_RE = _build_pattern(MEDICAL_KEYWORDS)

# ── Refusal messages ──────────────────────────────────────────────────────────

_MESSAGES: dict[str, str] = {
    "pii": (
        "For privacy and security reasons, I'm not able to look up or share "
        "personal information such as email addresses, phone numbers, "
        "identification numbers, or other personally identifiable data. "
        "Please rephrase your question without requesting personal details."
    ),
    "legal": (
        "I'm not qualified to give legal advice. Questions about laws, "
        "contracts, leases, or legal liability should be directed to a "
        "qualified legal professional. I can still help you understand "
        "general information from your uploaded documents."
    ),
    "medical": (
        "I'm not a medical professional and cannot provide medical advice. "
        "For questions about symptoms, diagnoses, treatments, or medication "
        "please consult a qualified healthcare provider. I can still help you "
        "find factual information from your uploaded documents."
    ),
}

# ── Public API ────────────────────────────────────────────────────────────────

_CHECKS: list[tuple[str, re.Pattern[str]]] = [
    ("pii",     _PII_RE),
    ("legal",   _LEGAL_RE),
    ("medical", _MEDICAL_RE),
]


def check_refusal(query: str) -> tuple[str, str] | tuple[None, None]:
    """
    Return (category, message) if the query should be refused,
    or (None, None) if it is safe to process.

    Categories: "pii" | "legal" | "medical"
    """
    for category, pattern in _CHECKS:
        if pattern.search(query):
            logger.info(
                "[refusal] category=%s  matched query=%r",
                category,
                query[:120],
            )
            return category, _MESSAGES[category]
    return None, None
