"""
utils.py
--------
Shared helper utilities.
"""

from modules.recommender import MODELS


# Tier badge colours used by both the CLI and the Streamlit app
TIER_COLORS = {
    "fast":      "#22c55e",   # green
    "balanced":  "#3b82f6",   # blue
    "powerful":  "#a855f7",   # purple
}

TIER_EMOJI = {
    "fast":      "⚡",
    "balanced":  "⚖️",
    "powerful":  "🧠",
}


def get_tier(model_key: str) -> str:
    return MODELS[model_key]["tier"]


def confidence_label(score: float) -> str:
    """Human-readable confidence tier."""
    if score >= 0.75:
        return "High"
    elif score >= 0.50:
        return "Medium"
    else:
        return "Low"


def format_cli_result(recommendation) -> str:
    """Pretty-print recommendation to terminal."""
    sep = "=" * 60
    lines = [
        sep,
        f"  RECOMMENDED MODEL : {recommendation.display_name}",
        f"  CONFIDENCE        : {recommendation.confidence:.1%}  ({confidence_label(recommendation.confidence)})",
        f"  RUNNER-UP         : {recommendation.runner_up}  ({recommendation.runner_up_confidence:.1%})",
        sep,
        "",
        "  EXPLANATION",
        "  " + "-" * 40,
    ]
    for line in recommendation.explanation.split("  \n"):
        # Strip markdown bold/italic markers for CLI output
        clean = line.replace("**", "").replace("*", "")
        lines.append("  " + clean)

    lines += [
        "",
        "  FEATURE SUMMARY",
        "  " + "-" * 40,
    ]
    for k, v in recommendation.feature_summary.items():
        lines.append(f"  {k:<25}: {v}")

    lines.append(sep)
    return "\n".join(lines)


def all_model_names() -> list:
    return [MODELS[k]["display_name"] for k in MODELS]
