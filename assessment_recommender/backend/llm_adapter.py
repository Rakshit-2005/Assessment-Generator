"""Lightweight JD parser — rule-based, free, no external API keys required.

This module extracts obvious skills and seniority level from a job description (JD).
It is intentionally simple so it runs offline without paid LLM services.
"""
import re
from typing import Dict, List

# Minimal, commonly-seen skills/tech tokens. Extend as needed.
COMMON_SKILLS = [
    'python','java','spring','sql','aws','docker','kubernetes','react','node','c++','c#',
    'go','rust','bash','git','jenkins','terraform','spark','hadoop','scala','ml','nlp',
    'nosql','postgres','mysql','mongodb','redis','aws','azure','gcp'
]


def extract_skills(jd: str) -> List[str]:
    text = (jd or '').lower()
    found = []
    for s in COMMON_SKILLS:
        if re.search(r"\b" + re.escape(s) + r"\b", text):
            found.append(s)
    return found


def extract_level(jd: str) -> str:
    t = (jd or '').lower()
    if re.search(r'\bsenior\b|\bsr\b|senior\s+ic', t):
        return 'Senior'
    if re.search(r'\bjunior\b|\bjr\b', t):
        return 'Junior'
    if re.search(r'\blead\b|\bmanager\b|\bprincipal\b', t):
        return 'Manager'
    return ''


def parse_jd(jd: str) -> Dict:
    """Return a dict with inferred `skills` (list) and `level` (string).

    Example: parse_jd('Senior backend engineer with Java, Spring, AWS') ->
    {'skills': ['java','spring','aws'], 'level': 'Senior'}
    """
    return {
        'skills': extract_skills(jd),
        'level': extract_level(jd)
    }
