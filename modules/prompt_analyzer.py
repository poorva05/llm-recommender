"""
prompt_analyzer.py
------------------
Extracts structured features from a raw user prompt.

Approach: Pure keyword/regex + heuristic scoring — no external NLP libraries needed.
This keeps the system lightweight and runnable on any laptop without model downloads.
Features extracted:
  - length_category  : short / medium / long
  - token_count      : approximate word count
  - keywords         : matched intent categories
  - complexity_score : 0.0–1.0 composite score
  - dominant_intent  : the top detected intent category
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Intent keyword bank
# Each category maps to a list of (pattern, weight) pairs.
# Weight reflects how strongly the keyword signals the intent.
# ---------------------------------------------------------------------------
INTENT_PATTERNS: Dict[str, List[tuple]] = {
    "coding": [
        (r"\b(code|coding|program|script|function|class|bug|debug|fix|implement|algorithm|api|sql|database|refactor|unittest|test)\b", 1.5),
        (r"\b(python|javascript|typescript|java|c\+\+|rust|go|bash|shell|html|css|react|django|flask)\b", 1.5),
        (r"\b(compile|syntax|error|exception|stack trace|import|library|package|module)\b", 1.2),
    ],
    "reasoning": [
        (r"\b(analyze|analysis|reason|compare|evaluate|pros and cons|trade-?off|decision|recommend|should i|best way|strategy|explain why|deduce|infer)\b", 1.5),
        (r"\b(logic|argument|thesis|hypothesis|evidence|conclusion|critique)\b", 1.2),
        (r"\b(math|mathemat|calculat|equation|formula|proof|derive|statistics|probability)\b", 1.3),
        (r"\b(entire history|full history|history of|evolution of|origins of)\b", 1.4),
    ],
    "creative_writing": [
        (r"\b(write|story|poem|essay|blog|creative|fiction|character|narrative|plot|dialogue|script|lyrics|novel)\b", 1.5),
        (r"\b(tone|style|voice|metaphor|imagery|description|imagine|invent|brainstorm)\b", 1.2),
    ],
    "summarization": [
        (r"\b(summarize|summary|tldr|brief|shorten|condense|key points|highlights|overview|outline|recap)\b", 1.5),
        (r"\b(extract|list the|bullet|main idea|important parts)\b", 1.1),
    ],
    "question_answering": [
        (r"\b(what is|what are|who is|where is|when did|how does|why does|tell me about|define|explain)\b", 1.3),
        (r"\b(fact|information|knowledge|history|science|geography|meaning of)\b", 1.1),
    ],
    "instruction_following": [
        (r"\b(translate|convert|format|list|generate|create|make|build|produce|fill|complete|draft)\b", 1.2),
        (r"\b(step by step|instructions|guide|how to|tutorial|walkthrough)\b", 1.4),
    ],
    "long_form": [
        (r"\b(detailed|comprehensive|in.depth|in depth|thorough|extensive|full|complete|research|report|document|whitepaper)\b", 1.5),
        (r"\b(all the|every|entire|everything about)\b", 1.1),
        (r"\b(\d{3,5})[- ]?word\b", 2.0),        # "2000-word" is a strong long-form signal
        (r"\b(\d+)[- ]?page\b", 1.8),              # "300-page PDF"
        (r"\b(across all|all section|full document|entire report)\b", 1.5),
    ],
}

# Complexity indicators — phrases that signal a harder / longer task
COMPLEXITY_BOOSTERS = [
    (r"\b(multiple|several|various|complex|advanced|expert|sophisticated)\b", 0.10),
    (r"\b(step by step|step-by-step|detailed explanation|in depth|comprehensive)\b", 0.12),
    (r"\b(compare and contrast|trade-?offs|nuanced|edge case|corner case)\b", 0.15),
    (r"\b(optimize|performance|scalab|architect|design pattern|best practice)\b", 0.12),
    (r"[?]{2,}", 0.05),          # multiple question marks → ambiguous / multi-part
    (r"[,;]{3,}", 0.08),          # lots of list separators → multi-item task
]

QUESTION_WORDS = re.compile(r"\b(who|what|where|when|why|how)\b", re.IGNORECASE)


@dataclass
class PromptFeatures:
    raw_prompt: str
    token_count: int = 0
    length_category: str = "short"        # short / medium / long
    intent_scores: Dict[str, float] = field(default_factory=dict)
    matched_keywords: List[str] = field(default_factory=list)
    dominant_intent: str = "question_answering"
    complexity_score: float = 0.0         # 0.0 (trivial) → 1.0 (very complex)
    question_count: int = 0
    has_code_block: bool = False


def analyze_prompt(prompt: str) -> PromptFeatures:
    """
    Main entry point.  Returns a PromptFeatures dataclass populated with
    all extracted signals.
    """
    features = PromptFeatures(raw_prompt=prompt)
    lower = prompt.lower()

    # ------------------------------------------------------------------
    # 1. Basic length features
    # ------------------------------------------------------------------
    tokens = prompt.split()
    features.token_count = len(tokens)

    if features.token_count <= 20:
        features.length_category = "short"
    elif features.token_count <= 80:
        features.length_category = "medium"
    else:
        features.length_category = "long"

    # ------------------------------------------------------------------
    # 2. Detect fenced code blocks (``` or indented 4-space blocks)
    # ------------------------------------------------------------------
    features.has_code_block = bool(re.search(r"```|^\s{4}\S", prompt, re.MULTILINE))

    # ------------------------------------------------------------------
    # 3. Count question marks / question words
    # ------------------------------------------------------------------
    features.question_count = prompt.count("?") + len(QUESTION_WORDS.findall(lower))

    # ------------------------------------------------------------------
    # 4. Score each intent category
    # ------------------------------------------------------------------
    intent_scores: Dict[str, float] = {k: 0.0 for k in INTENT_PATTERNS}
    matched_keywords: List[str] = []

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern, weight in patterns:
            hits = re.findall(pattern, lower)
            if hits:
                intent_scores[intent] += weight * len(hits)
                matched_keywords.extend(hits)

    # Boost coding score if code block found
    if features.has_code_block:
        intent_scores["coding"] += 2.0

    # Normalise scores to 0–1 range (relative to max possible per category)
    max_score = max(intent_scores.values()) if any(v > 0 for v in intent_scores.values()) else 1.0
    features.intent_scores = {k: round(v / max_score, 3) for k, v in intent_scores.items()}
    features.matched_keywords = list(set(matched_keywords))[:15]  # cap for display

    # Dominant intent = highest scoring category
    features.dominant_intent = max(features.intent_scores, key=features.intent_scores.get)

    # ------------------------------------------------------------------
    # 5. Complexity score
    # ------------------------------------------------------------------
    complexity = 0.0

    # Length contribution (max 0.30)
    length_map = {"short": 0.05, "medium": 0.15, "long": 0.30}
    complexity += length_map[features.length_category]

    # Booster phrases contribution (max ~0.50)
    for pattern, boost in COMPLEXITY_BOOSTERS:
        if re.search(pattern, lower):
            complexity += boost

    # Intent contribution: coding + reasoning are intrinsically harder
    hard_intents = {"coding", "reasoning", "long_form"}
    for intent in hard_intents:
        complexity += features.intent_scores.get(intent, 0.0) * 0.10

    # Word-count signals that explicitly ask for long output ("2000-word", "detailed report")
    word_count_requests = re.findall(r"\b(\d{3,5})[- ]?word", lower)
    if word_count_requests:
        complexity += 0.20

    # Explicit "detailed" / "comprehensive" creative or long-form request
    if (features.intent_scores.get("creative_writing", 0) > 0.3
            and features.intent_scores.get("long_form", 0) > 0.2):
        complexity += 0.18

    # Document / file / page count references
    if re.search(r"\b(\d+)[- ]?page|\b(pdf|document|report|whitepaper)\b", lower):
        complexity += 0.15

    # Clamp to [0.05, 1.0]
    features.complexity_score = round(min(max(complexity, 0.05), 1.0), 3)

    return features
