"""
app.py
------
Streamlit UI for the LLM Version Recommender System.

Run with:
    streamlit run app.py
"""

import streamlit as st
from modules.prompt_analyzer import analyze_prompt
from modules.recommender import recommend, MODELS
from modules.utils import TIER_COLORS, TIER_EMOJI, confidence_label

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean dark-accent theme with monospaced personality
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* ── Hero header ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 4px;
}

/* ── Card containers ── */
.rec-card {
    background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}
.metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.6rem;
    padding: 6px 20px;
    border-radius: 40px;
    margin-bottom: 12px;
}
.tier-fast    { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.tier-balanced{ background: #172554; color: #60a5fa; border: 1px solid #1e40af; }
.tier-powerful{ background: #2e1065; color: #d946ef; border: 1px solid #7e22ce; }

/* ── Confidence bar ── */
.conf-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 8px 0 16px 0;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    transition: width 0.5s ease;
}

/* ── Explanation text ── */
.explanation-text {
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.75;
}

/* ── Feature table ── */
.feat-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.88rem;
}
.feat-key { color: #64748b; }
.feat-val { color: #94a3b8; font-family: 'JetBrains Mono', monospace; }

/* ── Runner-up strip ── */
.runnerup {
    background: #0f172a;
    border: 1px dashed #334155;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 12px;
}

/* ── Intent chips ── */
.chip {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}
.chip-active {
    background: #312e81;
    color: #a5b4fc;
    border: 1px solid #4338ca;
}

/* ── Sidebar ── */
.sidebar-model {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
}
.sidebar-model-name { font-weight: 700; color: #e2e8f0; }
.sidebar-model-meta { color: #64748b; font-size: 0.75rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — model catalogue
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Model Catalogue")
    for key, info in MODELS.items():
        tier_color = TIER_COLORS[info["tier"]]
        emoji = TIER_EMOJI[info["tier"]]
        st.markdown(f"""
        <div class="sidebar-model">
            <div class="sidebar-model-name">{emoji} {info['display_name']}</div>
            <div class="sidebar-model-meta">
                Tier: <span style="color:{tier_color}">{info['tier'].capitalize()}</span>
                &nbsp;·&nbsp; {info['context_window']}<br>
                <i>{info['best_for']}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔧 How it works")
    st.markdown("""
- Extracts **length**, **intent keywords**, and **complexity score** from your prompt  
- Feeds features into a **Random Forest** classifier trained on synthetic data  
- A **rule-based override** layer catches obvious edge cases  
- Returns the best model + confidence score + explanation
""")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">🤖 LLM Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Paste your prompt — get the ideal model instantly.</p>', unsafe_allow_html=True)
st.markdown("")

# Example prompts
EXAMPLES = {
    "Simple question": "What is the capital of France?",
    "Code task": "Write a Python function to implement binary search on a sorted list.",
    "Complex analysis": "Compare microservices vs monolithic architecture for a fintech startup with 10M daily transactions. Cover DevOps maturity, team size, regulatory compliance, and cost at scale.",
    "Long document": "I have a 200-page research report on climate change. Please summarize all major findings, identify contradictions between sections, and produce a structured executive summary.",
    "Creative writing": "Write a detailed, emotionally resonant short story about an astronaut who discovers an ancient alien library on the dark side of the moon.",
}

col_ex, _ = st.columns([3, 1])
with col_ex:
    chosen_example = st.selectbox("💡 Load an example prompt:", ["— type your own —"] + list(EXAMPLES.keys()))

default_text = EXAMPLES.get(chosen_example, "")

prompt_input = st.text_area(
    label="Your prompt",
    value=default_text,
    height=140,
    placeholder="Describe what you want the AI to do...",
    label_visibility="collapsed",
)

analyse_btn = st.button("⚡  Analyse & Recommend", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------
if analyse_btn:
    if not prompt_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Analysing…"):
            features = analyze_prompt(prompt_input.strip())
            rec = recommend(features)

        tier = MODELS[rec.model_key]["tier"]

        st.markdown("---")

        # ── Top-level result ───────────────────────────────────────────────
        left, right = st.columns([2, 1])

        with left:
            st.markdown(f"""
            <div class="rec-card">
                <div class="metric-label">Recommended Model</div>
                <div class="model-badge tier-{tier}">
                    {TIER_EMOJI[tier]}&nbsp; {rec.display_name}
                </div>
                <div class="metric-label" style="margin-top:12px">Confidence — {confidence_label(rec.confidence)} ({rec.confidence:.1%})</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{rec.confidence*100:.1f}%"></div>
                </div>
                <div class="explanation-text">{rec.explanation.replace(chr(10), '<br>')}</div>
                <div class="runnerup">
                    🥈 Runner-up: <strong>{rec.runner_up}</strong>
                    &nbsp;({rec.runner_up_confidence:.1%} confidence)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("#### 📊 Feature Breakdown")
            for k, v in rec.feature_summary.items():
                st.markdown(f"""
                <div class="feat-row">
                    <span class="feat-key">{k}</span>
                    <span class="feat-val">{v}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🏷️ Detected Intents")
            # Show top 4 intents as chips
            sorted_intents = sorted(features.intent_scores.items(), key=lambda x: -x[1])
            chips_html = ""
            for intent, score in sorted_intents[:6]:
                active = "chip-active" if intent == features.dominant_intent else "chip"
                label = intent.replace("_", " ")
                chips_html += f'<span class="{active}">{label} {score:.2f}</span>'
            st.markdown(chips_html, unsafe_allow_html=True)

        # ── Intent score bar chart ─────────────────────────────────────────
        st.markdown("#### 📈 Intent Score Distribution")
        chart_data = {
            k.replace("_", " "): v
            for k, v in sorted(features.intent_scores.items(), key=lambda x: -x[1])
            if v > 0
        }
        if chart_data:
            st.bar_chart(chart_data, height=200)
        else:
            st.caption("No significant intent signals detected.")

        # ── Raw features expander ──────────────────────────────────────────
        with st.expander("🔬 Raw feature details"):
            st.json({
                "token_count": features.token_count,
                "length_category": features.length_category,
                "complexity_score": features.complexity_score,
                "dominant_intent": features.dominant_intent,
                "has_code_block": features.has_code_block,
                "question_count": features.question_count,
                "matched_keywords": features.matched_keywords,
                "intent_scores": features.intent_scores,
            })
