"""
generate_dataset.py
-------------------
Produces data/training_data.csv — 300 labelled records of prompt features.

Each row represents a realistic prompt scenario with hand-designed feature
distributions per model class, then lightly perturbed with Gaussian noise
to create diversity.  This is *synthetic-but-principled* data: the distributions
encode domain knowledge about each model's sweet spot.

Run:
    python generate_dataset.py
"""

import csv
import random
import math

random.seed(42)


# ---------------------------------------------------------------------------
# Column schema (must match _vectorize in recommender.py)
# ---------------------------------------------------------------------------
COLUMNS = [
    "token_count",
    "complexity_score",
    "coding",
    "reasoning",
    "creative_writing",
    "summarization",
    "question_answering",
    "instruction_following",
    "long_form",
    "has_code_block",
    "question_count_norm",
    "label",
]


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def jitter(v, sigma=0.07, lo=0.0, hi=1.0):
    """Add Gaussian noise and clamp."""
    return clamp(v + random.gauss(0, sigma), lo, hi)


def jitter_int(v, sigma=15, lo=1, hi=500):
    return max(lo, min(hi, int(v + random.gauss(0, sigma))))


# ---------------------------------------------------------------------------
# Per-class templates: (base_values_dict, count)
# ---------------------------------------------------------------------------
TEMPLATES = {

    # ── claude-haiku  ────────────────────────────────────────────────────────
    # Short, simple, low complexity.  Single-turn Q&A, quick summaries.
    "claude-haiku": {
        "n": 50,
        "token_count":           (10, 8),     # (mean, std)
        "complexity_score":      (0.08, 0.03),
        "coding":                (0.0, 0.05),
        "reasoning":             (0.0, 0.05),
        "creative_writing":      (0.1, 0.10),
        "summarization":         (0.5, 0.25),
        "question_answering":    (0.8, 0.15),
        "instruction_following": (0.1, 0.10),
        "long_form":             (0.0, 0.03),
        "has_code_block_prob":   0.02,
        "question_count_norm":   (0.3, 0.15),
    },

    # ── gpt-4o-mini  ─────────────────────────────────────────────────────────
    # Short-medium, mostly simple, some light summarization / chat.
    "gpt-4o-mini": {
        "n": 45,
        "token_count":           (22, 10),
        "complexity_score":      (0.14, 0.04),
        "coding":                (0.05, 0.08),
        "reasoning":             (0.05, 0.08),
        "creative_writing":      (0.25, 0.15),
        "summarization":         (0.55, 0.20),
        "question_answering":    (0.55, 0.20),
        "instruction_following": (0.30, 0.15),
        "long_form":             (0.03, 0.05),
        "has_code_block_prob":   0.05,
        "question_count_norm":   (0.25, 0.12),
    },

    # ── claude-sonnet  ───────────────────────────────────────────────────────
    # Moderate complexity, coding, instruction following, balanced writing.
    "claude-sonnet": {
        "n": 55,
        "token_count":           (40, 15),
        "complexity_score":      (0.42, 0.08),
        "coding":                (0.70, 0.20),
        "reasoning":             (0.35, 0.15),
        "creative_writing":      (0.15, 0.12),
        "summarization":         (0.10, 0.10),
        "question_answering":    (0.25, 0.15),
        "instruction_following": (0.65, 0.20),
        "long_form":             (0.08, 0.08),
        "has_code_block_prob":   0.45,
        "question_count_norm":   (0.20, 0.12),
    },

    # ── gpt-4o  ──────────────────────────────────────────────────────────────
    # Medium-high complexity, strong coding + reasoning, tool use.
    "gpt-4o": {
        "n": 50,
        "token_count":           (58, 18),
        "complexity_score":      (0.61, 0.08),
        "coding":                (0.75, 0.18),
        "reasoning":             (0.72, 0.15),
        "creative_writing":      (0.10, 0.10),
        "summarization":         (0.08, 0.08),
        "question_answering":    (0.20, 0.12),
        "instruction_following": (0.55, 0.18),
        "long_form":             (0.10, 0.10),
        "has_code_block_prob":   0.60,
        "question_count_norm":   (0.25, 0.12),
    },

    # ── gemini-1.5-pro  ──────────────────────────────────────────────────────
    # Very long prompts, document analysis, massive context.
    "gemini-1.5-pro": {
        "n": 50,
        "token_count":           (230, 60),
        "complexity_score":      (0.57, 0.08),
        "coding":                (0.15, 0.12),
        "reasoning":             (0.38, 0.15),
        "creative_writing":      (0.10, 0.10),
        "summarization":         (0.80, 0.15),
        "question_answering":    (0.28, 0.12),
        "instruction_following": (0.20, 0.12),
        "long_form":             (0.85, 0.10),
        "has_code_block_prob":   0.10,
        "question_count_norm":   (0.15, 0.10),
    },

    # ── claude-opus  ─────────────────────────────────────────────────────────
    # High complexity, nuanced reasoning, long creative writing, research.
    "claude-opus": {
        "n": 50,
        "token_count":           (92, 25),
        "complexity_score":      (0.83, 0.07),
        "coding":                (0.28, 0.18),
        "reasoning":             (0.88, 0.10),
        "creative_writing":      (0.65, 0.20),
        "summarization":         (0.12, 0.10),
        "question_answering":    (0.22, 0.12),
        "instruction_following": (0.35, 0.15),
        "long_form":             (0.50, 0.20),
        "has_code_block_prob":   0.15,
        "question_count_norm":   (0.40, 0.18),
    },
}


# ---------------------------------------------------------------------------
# Generate rows
# ---------------------------------------------------------------------------
rows = []

for label, cfg in TEMPLATES.items():
    n = cfg["n"]
    for _ in range(n):
        tc_mean, tc_std = cfg["token_count"]
        token_count = max(1, int(random.gauss(tc_mean, tc_std)))

        cs_mean, cs_std = cfg["complexity_score"]
        complexity = clamp(random.gauss(cs_mean, cs_std), 0.05, 1.0)

        def s(key):
            m, sd = cfg[key]
            return round(clamp(random.gauss(m, sd)), 3)

        coding         = s("coding")
        reasoning      = s("reasoning")
        creative       = s("creative_writing")
        summ           = s("summarization")
        qa             = s("question_answering")
        instr          = s("instruction_following")
        longf          = s("long_form")
        has_code       = 1 if random.random() < cfg["has_code_block_prob"] else 0
        qn_mean, qn_sd = cfg["question_count_norm"]
        q_norm         = round(clamp(random.gauss(qn_mean, qn_sd)), 3)

        rows.append({
            "token_count":           token_count,
            "complexity_score":      round(complexity, 3),
            "coding":                coding,
            "reasoning":             reasoning,
            "creative_writing":      creative,
            "summarization":         summ,
            "question_answering":    qa,
            "instruction_following": instr,
            "long_form":             longf,
            "has_code_block":        has_code,
            "question_count_norm":   q_norm,
            "label":                 label,
        })

# Shuffle so classes are interleaved
random.shuffle(rows)

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
OUT = "data/training_data.csv"
with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows → {OUT}")

# Quick sanity check
from collections import Counter
labels = [r["label"] for r in rows]
print("Class distribution:", dict(Counter(labels)))
