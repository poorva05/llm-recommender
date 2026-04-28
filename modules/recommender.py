"""
recommender.py
--------------
Recommends the most suitable LLM given a PromptFeatures object.

Design choice — Hybrid approach:
  1. A small synthetic dataset is created inside this file.
  2. A Random-Forest classifier (sklearn) is trained on it at startup.
  3. The classifier produces a probability distribution over model classes.
  4. A thin rule-based post-processor can override the ML result when
     highly specific signals exist (e.g., a very short trivial prompt).

Why ML instead of pure rules?
  - Rules alone struggle with mixed-intent prompts.
  - A trained classifier handles feature interactions automatically.
  - sklearn's RandomForest is CPU-only, trains in <1 s, zero external data.

Why keep rules at all?
  - Edge cases that are hard to capture in a small synthetic dataset
    (e.g., extremely short greetings → Haiku is the obvious answer).
  - Rules make the system interpretable and easy to audit.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from modules.prompt_analyzer import PromptFeatures


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
MODELS: Dict[str, Dict] = {
    "claude-haiku": {
        "display_name": "Claude Haiku",
        "tier": "fast",
        "strengths": ["simple Q&A", "summarization", "short tasks", "low latency"],
        "context_window": "200k tokens",
        "best_for": "quick, lightweight tasks",
    },
    "claude-sonnet": {
        "display_name": "Claude Sonnet",
        "tier": "balanced",
        "strengths": ["coding", "reasoning", "balanced writing", "instruction following"],
        "context_window": "200k tokens",
        "best_for": "everyday professional work",
    },
    "claude-opus": {
        "display_name": "Claude Opus",
        "tier": "powerful",
        "strengths": ["complex reasoning", "nuanced writing", "research", "multi-step tasks"],
        "context_window": "200k tokens",
        "best_for": "hard, multi-faceted problems",
    },
    "gpt-4o": {
        "display_name": "GPT-4o",
        "tier": "powerful",
        "strengths": ["vision", "code", "complex reasoning", "tool use"],
        "context_window": "128k tokens",
        "best_for": "multimodal & advanced reasoning",
    },
    "gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "tier": "fast",
        "strengths": ["simple tasks", "summarization", "classification", "chat"],
        "context_window": "128k tokens",
        "best_for": "cost-effective simple tasks",
    },
    "gemini-1.5-pro": {
        "display_name": "Gemini 1.5 Pro",
        "tier": "powerful",
        "strengths": ["very long context", "multimodal", "document analysis", "code"],
        "context_window": "1M tokens",
        "best_for": "massive documents & long context",
    },
}

MODEL_CLASSES = list(MODELS.keys())


# ---------------------------------------------------------------------------
# Synthetic training dataset
# ---------------------------------------------------------------------------
# Each row: [token_count, complexity_score, coding, reasoning, creative_writing,
#            summarization, question_answering, instruction_following, long_form,
#            has_code_block, question_count_norm]
# Label: model key from MODEL_CLASSES

def _build_training_data() -> Tuple[np.ndarray, List[str]]:
    """
    Returns (X, y) arrays.
    Data is hand-crafted to encode domain knowledge about when each model shines.
    Token counts and scores are normalised before training (see _vectorize).
    """
    rows = []
    labels = []

    def add(token_count, complexity, coding, reasoning, creative, summ, qa, instr, longf,
            code_block, q_count, label):
        rows.append([token_count, complexity, coding, reasoning, creative, summ, qa, instr,
                     longf, int(code_block), q_count])
        labels.append(label)

    # ---- claude-haiku: short, simple, low complexity ----
    add(5,  0.06, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, False, 1, "claude-haiku")
    add(8,  0.07, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, False, 0, "claude-haiku")
    add(12, 0.08, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, False, 2, "claude-haiku")
    add(10, 0.07, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, False, 0, "claude-haiku")
    add(15, 0.10, 0.0, 0.0, 0.0, 0.0, 0.9, 0.2, 0.0, False, 1, "claude-haiku")
    add(6,  0.06, 0.0, 0.0, 0.3, 0.0, 0.5, 0.0, 0.0, False, 0, "claude-haiku")

    # ---- gpt-4o-mini: short-medium, simple, no heavy reasoning ----
    add(18, 0.12, 0.0, 0.0, 0.2, 0.5, 0.5, 0.3, 0.0, False, 1, "gpt-4o-mini")
    add(22, 0.15, 0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, False, 0, "gpt-4o-mini")
    add(25, 0.14, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.0, False, 2, "gpt-4o-mini")
    add(20, 0.13, 0.0, 0.0, 0.4, 0.0, 0.4, 0.3, 0.0, False, 0, "gpt-4o-mini")

    # ---- claude-sonnet: medium complexity, coding, instruction following ----
    add(35, 0.40, 1.0, 0.3, 0.0, 0.0, 0.2, 0.5, 0.0, False, 1, "claude-sonnet")
    add(45, 0.50, 0.8, 0.5, 0.0, 0.0, 0.2, 0.6, 0.0, True,  0, "claude-sonnet")
    add(30, 0.35, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, False, 0, "claude-sonnet")
    add(50, 0.45, 0.5, 0.4, 0.0, 0.2, 0.0, 0.5, 0.0, False, 1, "claude-sonnet")
    add(40, 0.42, 1.0, 0.2, 0.0, 0.0, 0.0, 0.7, 0.0, True,  0, "claude-sonnet")
    add(28, 0.38, 0.0, 0.5, 0.2, 0.0, 0.4, 0.5, 0.0, False, 2, "claude-sonnet")

    # ---- gpt-4o: medium-high, coding + reasoning, tool use ----
    add(55, 0.60, 0.8, 0.7, 0.0, 0.0, 0.3, 0.5, 0.0, True,  1, "gpt-4o")
    add(60, 0.65, 0.7, 0.8, 0.0, 0.0, 0.2, 0.6, 0.0, True,  0, "gpt-4o")
    add(45, 0.55, 1.0, 0.6, 0.0, 0.0, 0.1, 0.7, 0.0, True,  2, "gpt-4o")
    add(70, 0.62, 0.6, 0.9, 0.0, 0.0, 0.3, 0.4, 0.0, False, 1, "gpt-4o")
    add(50, 0.58, 0.9, 0.5, 0.0, 0.0, 0.2, 0.8, 0.0, True,  0, "gpt-4o")

    # ---- gemini-1.5-pro: very long prompts, document analysis ----
    add(200, 0.55, 0.2, 0.4, 0.0, 0.8, 0.3, 0.2, 1.0, False, 0, "gemini-1.5-pro")
    add(300, 0.60, 0.0, 0.3, 0.0, 1.0, 0.2, 0.2, 1.0, False, 1, "gemini-1.5-pro")
    add(180, 0.52, 0.0, 0.5, 0.2, 0.7, 0.4, 0.0, 1.0, False, 0, "gemini-1.5-pro")
    add(250, 0.58, 0.3, 0.3, 0.0, 0.9, 0.2, 0.3, 1.0, False, 0, "gemini-1.5-pro")

    # ---- claude-opus: high complexity, nuanced reasoning, long creative ----
    add(80,  0.80, 0.3, 1.0, 0.5, 0.0, 0.3, 0.4, 0.3, False, 2, "claude-opus")
    add(100, 0.85, 0.2, 1.0, 0.8, 0.0, 0.2, 0.3, 0.5, False, 1, "claude-opus")
    add(90,  0.82, 0.5, 0.9, 0.3, 0.0, 0.2, 0.5, 0.4, False, 2, "claude-opus")
    add(70,  0.78, 0.0, 1.0, 0.9, 0.0, 0.3, 0.2, 0.4, False, 1, "claude-opus")
    add(120, 0.90, 0.4, 0.8, 0.7, 0.0, 0.2, 0.4, 0.6, False, 3, "claude-opus")
    add(85,  0.83, 0.0, 0.9, 0.6, 0.3, 0.3, 0.3, 0.5, False, 2, "claude-opus")

    return np.array(rows, dtype=float), labels


def _vectorize(features: PromptFeatures) -> np.ndarray:
    """Convert PromptFeatures → model input vector (same schema as training data)."""
    intents = features.intent_scores
    return np.array([[
        features.token_count,
        features.complexity_score,
        intents.get("coding", 0.0),
        intents.get("reasoning", 0.0),
        intents.get("creative_writing", 0.0),
        intents.get("summarization", 0.0),
        intents.get("question_answering", 0.0),
        intents.get("instruction_following", 0.0),
        intents.get("long_form", 0.0),
        int(features.has_code_block),
        min(features.question_count / 5.0, 1.0),   # normalise question count
    ]], dtype=float)


# ---------------------------------------------------------------------------
# Recommendation result
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    model_key: str
    display_name: str
    confidence: float          # 0.0–1.0
    explanation: str
    runner_up: str             # second-best model
    runner_up_confidence: float
    feature_summary: Dict


# ---------------------------------------------------------------------------
# Classifier (trained once at import time)
# ---------------------------------------------------------------------------
_X, _y = _build_training_data()
_le = LabelEncoder().fit(_y)
_clf = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=6)
_clf.fit(_X, _le.transform(_y))


# ---------------------------------------------------------------------------
# Rule-based override layer
# ---------------------------------------------------------------------------
def _rule_override(features: PromptFeatures) -> str | None:
    """
    Returns a model key if a high-confidence rule fires, else None.
    These rules catch obvious edge-cases that a small dataset might miss.
    """
    # Tiny greetings / single-word prompts → Haiku
    if features.token_count <= 5 and features.complexity_score < 0.12:
        return "claude-haiku"

    # Very long prompt (>150 tokens) that's mostly summarization → Gemini
    if features.token_count > 150 and features.intent_scores.get("long_form", 0) > 0.5:
        return "gemini-1.5-pro"

    # Explicit long document reference → Gemini
    if (features.intent_scores.get("long_form", 0) > 0.7
            and features.intent_scores.get("summarization", 0) > 0.3):
        return "gemini-1.5-pro"

    # Long creative writing request → Claude Opus
    if (features.intent_scores.get("creative_writing", 0) > 0.5
            and features.complexity_score > 0.30):
        return "claude-opus"

    # Deep reasoning / in-depth explanation → Opus
    if (features.intent_scores.get("reasoning", 0) > 0.4
            and features.intent_scores.get("long_form", 0) > 0.3):
        return "claude-opus"

    # Coding (debug, fix, implement) any detectable coding signal → Sonnet
    if features.intent_scores.get("coding", 0) > 0.3:
        if features.complexity_score >= 0.55:
            return "gpt-4o"
        return "claude-sonnet"

    # Code block present + high coding intent + medium complexity → Sonnet
    if (features.has_code_block
            and features.intent_scores.get("coding", 0) > 0.3):
        return "claude-sonnet"

    # Medium-high complexity reasoning → Claude Sonnet / GPT-4o
    if features.complexity_score >= 0.55 and features.intent_scores.get("reasoning", 0) > 0.3:
        return "gpt-4o"

    return None


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------
def _build_explanation(model_key: str, features: PromptFeatures, confidence: float) -> str:
    model = MODELS[model_key]
    lines = []

    lines.append(
        f"**{model['display_name']}** is recommended because it excels at "
        f"*{model['best_for']}*."
    )

    # Dominant intent rationale
    intent = features.dominant_intent.replace("_", " ")
    lines.append(f"Your prompt is primarily a **{intent}** task.")

    # Length rationale
    length_msg = {
        "short":  "The prompt is concise, so a fast/light model is appropriate.",
        "medium": "The prompt has moderate length, balancing detail and cost.",
        "long":   "The prompt is long — a high-context model handles this best.",
    }
    lines.append(length_msg[features.length_category])

    # Complexity rationale
    if features.complexity_score < 0.25:
        lines.append("Complexity is **low** — no heavy reasoning required.")
    elif features.complexity_score < 0.55:
        lines.append("Complexity is **moderate** — a capable mid-tier model fits.")
    elif features.complexity_score < 0.75:
        lines.append("Complexity is **high** — a powerful model is warranted.")
    else:
        lines.append("Complexity is **very high** — only a top-tier model will do.")

    # Strength match
    strengths = ", ".join(model["strengths"][:3])
    lines.append(f"Key strengths matched: {strengths}.")

    # Keywords
    if features.matched_keywords:
        kw = ", ".join(features.matched_keywords[:6])
        lines.append(f"Detected keywords: *{kw}*.")

    return "  \n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def recommend(features: PromptFeatures) -> Recommendation:
    """
    Given extracted prompt features, return a Recommendation.
    """
    # 1. Rule override
    override = _rule_override(features)

    # 2. ML prediction
    x = _vectorize(features)
    proba = _clf.predict_proba(x)[0]          # shape: (n_classes,)
    class_indices = np.argsort(proba)[::-1]   # descending
    top_idx = class_indices[0]
    second_idx = class_indices[1]

    ml_model = _le.inverse_transform([top_idx])[0]
    ml_conf = float(proba[top_idx])

    # Runner-up: find the best model that isn't the primary
    runner_up_idx = None
    for idx in class_indices[1:]:
        candidate = _le.inverse_transform([idx])[0]
        if candidate != ml_model:
            runner_up_idx = idx
            break
    runner_up_idx = runner_up_idx if runner_up_idx is not None else class_indices[1]
    runner_up = _le.inverse_transform([runner_up_idx])[0]
    runner_up_conf = float(proba[runner_up_idx])

    # Apply rule override if present
    final_model = override if override else ml_model
    # If overridden, give it slightly elevated confidence
    final_conf = min(ml_conf + 0.10, 0.98) if override else ml_conf

    # Ensure runner-up differs from final recommendation
    if runner_up == final_model:
        for idx in class_indices:
            candidate = _le.inverse_transform([idx])[0]
            if candidate != final_model:
                runner_up = candidate
                runner_up_conf = float(proba[idx])
                break

    explanation = _build_explanation(final_model, features, final_conf)

    feature_summary = {
        "Token count": features.token_count,
        "Length": features.length_category,
        "Complexity": f"{features.complexity_score:.2f}",
        "Dominant intent": features.dominant_intent.replace("_", " "),
        "Code block detected": str(features.has_code_block),
        "Question signals": features.question_count,
    }

    return Recommendation(
        model_key=final_model,
        display_name=MODELS[final_model]["display_name"],
        confidence=round(final_conf, 3),
        explanation=explanation,
        runner_up=MODELS[runner_up]["display_name"],
        runner_up_confidence=round(runner_up_conf, 3),
        feature_summary=feature_summary,
    )
