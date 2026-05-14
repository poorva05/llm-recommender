"""
app.py  ·  LLM Version Recommender System
Streamlit UI — terminal-intelligence aesthetic.
Run:  streamlit run app.py
"""

import streamlit as st
from modules.prompt_analyzer import analyze_prompt
from modules.recommender import recommend, MODELS
from modules.utils import confidence_label

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Recommender",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── per-model visual identity ─────────────────────────────────────────────────
MODEL_META = {
    "claude-haiku":   {"icon": "⚡", "accent": "#10b981", "glow": "rgba(16,185,129,0.20)",  "label": "FAST"},
    "claude-sonnet":  {"icon": "⚖️", "accent": "#3b82f6", "glow": "rgba(59,130,246,0.20)",  "label": "BALANCED"},
    "claude-opus":    {"icon": "🧠", "accent": "#8b5cf6", "glow": "rgba(139,92,246,0.20)",   "label": "POWERFUL"},
    "gpt-4o":         {"icon": "🔬", "accent": "#f59e0b", "glow": "rgba(245,158,11,0.20)",   "label": "POWERFUL"},
    "gpt-4o-mini":    {"icon": "💨", "accent": "#06b6d4", "glow": "rgba(6,182,212,0.20)",    "label": "FAST"},
    "gemini-1.5-pro": {"icon": "📚", "accent": "#ec4899", "glow": "rgba(236,72,153,0.20)",   "label": "POWERFUL"},
}

INTENT_ICONS = {
    "coding": "💻", "reasoning": "🧩", "creative_writing": "✍️",
    "summarization": "📋", "question_answering": "❓",
    "instruction_following": "📌", "long_form": "📄",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp { background-color:#060912!important; color:#e2e8f0; font-family:'Space Grotesk',sans-serif; }
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 2.5rem 4rem!important;max-width:1200px}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:#0d1117}::-webkit-scrollbar-thumb{background:#1e293b;border-radius:3px}

/* HEADER */
.llm-header{display:flex;align-items:flex-end;gap:18px;padding:8px 0 28px;border-bottom:1px solid #0f1f3d;margin-bottom:30px}
.llm-logo{font-size:2.8rem;line-height:1;filter:drop-shadow(0 0 14px rgba(99,179,237,.6))}
.llm-h1{font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:#f0f9ff;letter-spacing:-.5px;margin:0 0 2px}
.llm-sub{font-size:.82rem;color:#334155;margin:0;font-family:'Space Mono',monospace}

/* INPUT */
.input-label{font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:2.5px;text-transform:uppercase;color:#334155;margin-bottom:8px}
.stTextArea textarea{background:#0d1117!important;border:1px solid #1e293b!important;border-radius:10px!important;color:#e2e8f0!important;font-family:'Space Mono',monospace!important;font-size:.88rem!important;line-height:1.7!important;padding:14px 16px!important;transition:border-color .2s,box-shadow .2s!important}
.stTextArea textarea:focus{border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.15)!important;outline:none!important}
.stSelectbox>div>div{background:#0d1117!important;border:1px solid #1e293b!important;border-radius:8px!important;color:#94a3b8!important;font-size:.84rem!important}

/* BUTTON */
.stButton>button{background:linear-gradient(135deg,#1d4ed8,#4f46e5)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:2px!important;padding:10px 24px!important;text-transform:uppercase!important;box-shadow:0 4px 20px rgba(79,70,229,.35)!important;transition:all .2s!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 28px rgba(79,70,229,.5)!important}

/* HERO RESULT CARD */
.result-hero{border-radius:16px;padding:26px 30px;position:relative;overflow:hidden;background:linear-gradient(145deg,#0a0f1e,#060912)}
.hero-glow{position:absolute;top:-50px;right:-50px;width:240px;height:240px;border-radius:50%;filter:blur(70px);pointer-events:none}
.hero-eyebrow{font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:3px;text-transform:uppercase;color:#334155;margin-bottom:10px}
.hero-model{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;line-height:1.1;margin-bottom:8px}
.hero-tier{display:inline-block;border-radius:4px;padding:2px 10px;font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px}
.gauge-label{font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:2.5px;text-transform:uppercase;color:#334155;margin-bottom:6px}
.gauge-row{display:flex;align-items:center;gap:14px;margin-bottom:22px}
.gauge-track{flex:1;height:5px;background:#1e293b;border-radius:999px;overflow:hidden}
.gauge-fill{height:100%;border-radius:999px}
.gauge-pct{font-family:'Space Mono',monospace;font-size:1.05rem;font-weight:700;min-width:48px;text-align:right}
.expl-block{background:rgba(15,23,42,.9);border-radius:0 8px 8px 0;padding:16px 20px;font-size:.86rem;line-height:1.82;color:#94a3b8}
.runner-pill{display:inline-flex;align-items:center;gap:8px;background:#0d1117;border:1px dashed #1e293b;border-radius:999px;padding:5px 16px;font-size:.76rem;color:#475569;margin-top:16px;font-family:'Space Mono',monospace}

/* STATS PANEL */
.stats-panel{background:#0a0f1e;border:1px solid #0f1f3d;border-radius:12px;padding:18px}
.panel-title{font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:3px;text-transform:uppercase;color:#334155;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #0f1f3d}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #0a0f1e}
.stat-row:last-child{border-bottom:none}
.stat-key{font-size:.78rem;color:#334155}
.stat-val{font-family:'Space Mono',monospace;font-size:.76rem;color:#64748b;background:#0d1117;padding:2px 8px;border-radius:4px}

/* COMPLEXITY RING */
.cx-row{display:flex;align-items:center;gap:18px;background:#0a0f1e;border:1px solid #0f1f3d;border-radius:12px;padding:16px 20px;margin-bottom:12px}
.cx-ring{position:relative;width:72px;height:72px;flex-shrink:0}
.cx-ring svg{transform:rotate(-90deg)}
.ring-bg{fill:none;stroke:#1e293b;stroke-width:6}
.ring-fg{fill:none;stroke-width:6;stroke-linecap:round}
.ring-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring-pct{font-family:'Space Mono',monospace;font-size:.95rem;font-weight:700;line-height:1}
.ring-sub{font-size:.52rem;letter-spacing:1px;text-transform:uppercase;color:#334155;margin-top:2px}
.cx-info h4{font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:#334155;margin:0 0 4px}
.cx-info p{font-size:.83rem;color:#64748b;margin:0}

/* INTENT CARDS */
.sec-label{font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:3px;text-transform:uppercase;color:#334155;margin:26px 0 12px;display:flex;align-items:center;gap:10px}
.sec-label::after{content:'';flex:1;height:1px;background:#0f1f3d}
.intent-card{background:#0a0f1e;border:1px solid #0f1f3d;border-radius:8px;padding:10px 12px;margin-bottom:0}
.intent-card.dom{border-color:var(--ia)}
.ic-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.ic-name{font-size:.76rem;color:#475569;text-transform:capitalize}
.intent-card.dom .ic-name{color:var(--ia)}
.ic-score{font-family:'Space Mono',monospace;font-size:.68rem;color:#334155}
.intent-card.dom .ic-score{color:var(--ia)}
.ib-track{height:3px;background:#1e293b;border-radius:999px;overflow:hidden}
.ib-fill{height:100%;border-radius:999px;background:#1e293b}
.intent-card.dom .ib-fill{background:var(--ia)}

/* MODEL GRID */
.model-mini{background:#0a0f1e;border:1px solid #0f1f3d;border-radius:10px;padding:12px 14px;transition:border-color .2s,transform .2s}
.model-mini.rec{border-color:var(--ma);transform:translateY(-2px)}
.mm-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.mm-icon{font-size:1.05rem;line-height:1}
.mm-name{font-size:.82rem;font-weight:600;color:#64748b}
.model-mini.rec .mm-name{color:var(--ma)}
.mm-tier{display:inline-block;font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#334155;background:#0d1117;padding:1px 6px;border-radius:3px;margin-bottom:5px}
.mm-ctx{font-family:'Space Mono',monospace;font-size:.66rem;color:#334155}
.mm-best{font-size:.72rem;color:#475569;margin-top:3px;line-height:1.35}

/* KW TAGS */
.kw-wrap{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}
.kw-tag{background:#0d1117;border:1px solid #1e293b;border-radius:4px;padding:2px 9px;font-family:'Space Mono',monospace;font-size:.68rem;color:#475569}
</style>
""", unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="llm-header">
  <div class="llm-logo">🚀</div>
  <div>
    <div class="llm-h1">LLM Recommender</div>
    <div class="llm-sub">// paste your prompt · get the ideal model · understand why</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── INPUT ─────────────────────────────────────────────────────────────────────
EXAMPLES = {
    "— type your own —": "",
    "Simple Q&A": "What is the capital of France?",
    "Code task": "Write a Python function to implement binary search on a sorted list and return the index.",
    "Debug request": "Fix this bug in my React component: useState is not re-rendering after an async update.",
    "Complex reasoning": "Compare microservices vs monolithic architecture for a fintech startup with 10M daily transactions. Cover DevOps maturity, team size, compliance, and cost.",
    "Creative long-form": "Write a detailed, emotionally resonant 2000-word story about an astronaut who finds an ancient alien library on the dark side of the moon.",
    "Long document": "I have a 300-page PDF of financial reports. Summarize key trends across all sections and flag any contradictions.",
}

col_in, col_btn = st.columns([4, 1])
with col_in:
    st.markdown('<div class="input-label">// your prompt</div>', unsafe_allow_html=True)
    chosen = st.selectbox("ex", list(EXAMPLES.keys()), label_visibility="collapsed")
    prompt_input = st.text_area(
        "p", value=EXAMPLES[chosen], height=118,
        placeholder="Describe what you want the AI to do...",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown('<div style="height:46px"></div>', unsafe_allow_html=True)
    analyse_btn = st.button("▶  ANALYSE", use_container_width=True)


# ── RESULTS ───────────────────────────────────────────────────────────────────
if analyse_btn:
    if not prompt_input.strip():
        st.warning("Enter a prompt first.")
        st.stop()

    with st.spinner("Analysing..."):
        features = analyze_prompt(prompt_input.strip())
        rec      = recommend(features)

    meta      = MODEL_META.get(rec.model_key, MODEL_META["claude-sonnet"])
    accent    = meta["accent"]
    glow      = meta["glow"]
    icon      = meta["icon"]
    tier_lbl  = meta["label"]
    conf_pct  = int(rec.confidence * 100)

    # ── separator ──
    st.markdown('<hr style="border:none;border-top:1px solid #0f1f3d;margin:20px 0">', unsafe_allow_html=True)

    # ── ROW 1: hero + stats ───────────────────────────────────────────────
    c1, c2 = st.columns([5, 3], gap="large")

    with c1:
        expl_html = rec.explanation.replace("**","<b>",1).replace("**","</b>",1)
        for _ in range(6):
            expl_html = expl_html.replace("**","<b>",1).replace("**","</b>",1)
        expl_html = expl_html.replace("*","<i>").replace("*","</i>")
        expl_html = expl_html.replace("  \n","<br>")

        st.markdown(f"""
        <div class="result-hero" style="border:1px solid {accent}22">
          <div class="hero-glow" style="background:{glow}"></div>
          <div class="hero-eyebrow">// recommended model</div>
          <div class="hero-model" style="color:{accent};text-shadow:0 0 30px {glow}">{icon}&nbsp; {rec.display_name}</div>
          <div class="hero-tier" style="background:{glow};color:{accent};border:1px solid {accent}55">{tier_lbl}</div>
          <div class="gauge-label">confidence</div>
          <div class="gauge-row">
            <div class="gauge-track">
              <div class="gauge-fill" style="width:{conf_pct}%;background:linear-gradient(90deg,{accent},{accent}88);box-shadow:0 0 8px {glow}"></div>
            </div>
            <div class="gauge-pct" style="color:{accent}">{conf_pct}%</div>
          </div>
          <div class="expl-block" style="border-left:3px solid {accent}">{expl_html}</div>
          <div class="runner-pill">🥈&nbsp;<span style="color:#64748b">Runner-up:</span>&nbsp;{rec.runner_up}&nbsp;·&nbsp;{int(rec.runner_up_confidence*100)}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # complexity ring
        r, cx2, cy2 = 30, 36, 36
        circ  = 2 * 3.14159 * r
        dash  = circ * features.complexity_score
        gap   = circ - dash
        cpct  = int(features.complexity_score * 100)
        rcol  = "#10b981" if cpct < 30 else "#f59e0b" if cpct < 60 else "#ef4444"
        cword = ("Trivial" if cpct < 20 else "Low" if cpct < 40 else
                 "Medium"  if cpct < 60 else "High" if cpct < 80 else "Very High")

        st.markdown(f"""
        <div class="cx-row">
          <div class="cx-ring">
            <svg width="72" height="72" viewBox="0 0 72 72">
              <circle class="ring-bg" cx="{cx2}" cy="{cy2}" r="{r}"/>
              <circle class="ring-fg" cx="{cx2}" cy="{cy2}" r="{r}"
                stroke="{rcol}" stroke-dasharray="{dash:.1f} {gap:.1f}"
                style="filter:drop-shadow(0 0 4px {rcol})"/>
            </svg>
            <div class="ring-inner">
              <span class="ring-pct" style="color:{rcol}">{cpct}%</span>
              <span class="ring-sub">cmplx</span>
            </div>
          </div>
          <div class="cx-info">
            <h4>Complexity</h4>
            <p><b style="color:{rcol}">{cword}</b><br>
            {features.token_count} tokens · {features.length_category}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="stats-panel"><div class="panel-title">// feature breakdown</div>', unsafe_allow_html=True)
        for k, v in rec.feature_summary.items():
            st.markdown(f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 2: intent analysis ────────────────────────────────────────────
    st.markdown('<div class="sec-label">// intent analysis</div>', unsafe_allow_html=True)

    top_intents = [(k, v) for k, v in
                   sorted(features.intent_scores.items(), key=lambda x: -x[1]) if v > 0][:6]

    cols_i = st.columns(3)
    for idx, (ik, score) in enumerate(top_intents):
        dom   = (ik == features.dominant_intent)
        iicon = INTENT_ICONS.get(ik, "🔹")
        lbl   = ik.replace("_", " ").title()
        bw    = int(score * 100)
        ia    = accent if dom else "#1e293b"
        with cols_i[idx % 3]:
            st.markdown(f"""
            <div class="intent-card {'dom' if dom else ''}" style="--ia:{ia};margin-bottom:8px">
              <div class="ic-top">
                <span class="ic-name">{iicon} {lbl}</span>
                <span class="ic-score">{score:.2f}</span>
              </div>
              <div class="ib-track"><div class="ib-fill" style="width:{bw}%"></div></div>
            </div>
            """, unsafe_allow_html=True)

    # keywords
    if features.matched_keywords:
        st.markdown('<div class="sec-label">// matched keywords</div>', unsafe_allow_html=True)
        tags = "".join(f'<span class="kw-tag">{kw}</span>' for kw in features.matched_keywords[:14])
        st.markdown(f'<div class="kw-wrap">{tags}</div>', unsafe_allow_html=True)

    # ── ROW 3: model grid ─────────────────────────────────────────────────
    st.markdown('<div class="sec-label">// all models</div>', unsafe_allow_html=True)

    model_items = list(MODELS.items())
    m_cols = st.columns(3)
    for idx, (mk, info) in enumerate(model_items):
        mm    = MODEL_META.get(mk, {})
        is_r  = (mk == rec.model_key)
        ma    = mm.get("accent", "#475569")
        mg    = mm.get("glow",   "transparent")
        mi    = mm.get("icon",   "🤖")
        ml    = mm.get("label",  info["tier"].upper())
        chk   = " ✓" if is_r else ""
        with m_cols[idx % 3]:
            st.markdown(f"""
            <div class="model-mini {'rec' if is_r else ''}" style="--ma:{ma};--mg:{mg};{'box-shadow:0 0 16px '+mg if is_r else ''};margin-bottom:8px">
              <div class="mm-head">
                <span class="mm-icon">{mi}</span>
                <span class="mm-name">{info['display_name']}{chk}</span>
              </div>
              <div class="mm-tier">{ml}</div>
              <div class="mm-ctx">{info['context_window']}</div>
              <div class="mm-best">{info['best_for']}</div>
            </div>
            """, unsafe_allow_html=True)

    # raw JSON
    with st.expander("// raw feature dump"):
        st.json({
            "token_count": features.token_count, "length_category": features.length_category,
            "complexity_score": features.complexity_score, "dominant_intent": features.dominant_intent,
            "has_code_block": features.has_code_block, "question_count": features.question_count,
            "matched_keywords": features.matched_keywords, "intent_scores": features.intent_scores,
        })

else:
    # ── idle state ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">// available models</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.83rem;color:#334155;margin-bottom:18px;font-family:Space Mono,monospace">enter a prompt above · get instant recommendation</p>', unsafe_allow_html=True)

    model_items = list(MODELS.items())
    m_cols = st.columns(3)
    for idx, (mk, info) in enumerate(model_items):
        mm = MODEL_META.get(mk, {})
        ma = mm.get("accent","#475569")
        mg = mm.get("glow","transparent")
        mi = mm.get("icon","🤖")
        ml = mm.get("label", info["tier"].upper())
        strengths = " · ".join(info["strengths"][:3])
        with m_cols[idx % 3]:
            st.markdown(f"""
            <div class="model-mini" style="--ma:{ma};--mg:{mg};margin-bottom:8px">
              <div class="mm-head">
                <span class="mm-icon">{mi}</span>
                <span class="mm-name" style="color:{ma}">{info['display_name']}</span>
              </div>
              <div class="mm-tier" style="color:{ma};background:color-mix(in srgb,{ma} 12%,transparent);border:1px solid color-mix(in srgb,{ma} 35%,transparent)">{ml}</div>
              <div class="mm-ctx">{info['context_window']}</div>
              <div class="mm-best">{strengths}</div>
            </div>
            """, unsafe_allow_html=True)
