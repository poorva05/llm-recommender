"""
model_evaluation.py
-------------------
Trains the Random Forest on data/training_data.csv and produces:
  - Accuracy, Macro F1, Per-class Precision / Recall / F1
  - Confusion matrix  → outputs/confusion_matrix.png
  - Feature importance → outputs/feature_importance.png
  - Full classification report printed to stdout
  - Results saved to outputs/evaluation_report.txt

Run:
    python model_evaluation.py
"""

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/training_data.csv")
OUT_DIR   = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
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
]

FEATURE_LABELS = [
    "Token Count",
    "Complexity Score",
    "Intent: Coding",
    "Intent: Reasoning",
    "Intent: Creative Writing",
    "Intent: Summarization",
    "Intent: Question Answering",
    "Intent: Instruction Follow",
    "Intent: Long Form",
    "Has Code Block",
    "Question Count (norm)",
]

# Nicer labels for plots
MODEL_DISPLAY = {
    "claude-haiku":    "Claude\nHaiku",
    "claude-sonnet":   "Claude\nSonnet",
    "claude-opus":     "Claude\nOpus",
    "gpt-4o":          "GPT-4o",
    "gpt-4o-mini":     "GPT-4o\nMini",
    "gemini-1.5-pro":  "Gemini\n1.5 Pro",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("LLM Recommender — Model Evaluation")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset loaded: {len(df)} records, {df['label'].nunique()} classes")
print("\nClass distribution:")
print(df["label"].value_counts().to_string())

X = df[FEATURE_COLS].values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_

# ---------------------------------------------------------------------------
# Train / test split (stratified 80 / 20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

print(f"\nTrain size: {len(X_train)}   Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
)
clf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\n{'─'*40}")
print(f"  Hold-out Test Metrics")
print(f"{'─'*40}")
print(f"  Accuracy         : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Macro F1-Score   : {macro_f1:.4f}")
print(f"  Weighted F1-Score: {weighted_f1:.4f}")
print(f"{'─'*40}")

# ---------------------------------------------------------------------------
# 5-Fold Stratified Cross-Validation (whole dataset)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_acc = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
cv_f1  = cross_val_score(clf, X, y_enc, cv=cv, scoring="f1_macro")

print(f"\n  5-Fold Cross-Validation")
print(f"{'─'*40}")
print(f"  CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  CV Macro F1  : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")
print(f"  Fold accuracies: {[f'{v:.3f}' for v in cv_acc]}")
print(f"{'─'*40}")

# ---------------------------------------------------------------------------
# Per-class Classification Report
# ---------------------------------------------------------------------------
report_str = classification_report(
    y_test, y_pred,
    target_names=classes,
    digits=3,
)
print(f"\nPer-Class Classification Report:\n")
print(report_str)

# ---------------------------------------------------------------------------
# Save text report
# ---------------------------------------------------------------------------
report_txt = textwrap.dedent(f"""
LLM Recommender — Evaluation Report
=====================================

Dataset : {DATA_PATH}
Records : {len(df)}
Classes : {list(classes)}

Hold-out Test Split (80/20, stratified)
-----------------------------------------
Accuracy          : {acc:.4f}  ({acc*100:.1f}%)
Macro F1-Score    : {macro_f1:.4f}
Weighted F1-Score : {weighted_f1:.4f}

5-Fold Stratified Cross-Validation
-----------------------------------------
CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}
CV Macro F1  : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}
Fold accuracies: {[f'{v:.3f}' for v in cv_acc]}

Per-Class Classification Report
-----------------------------------------
{report_str}
""").strip()

with open(OUT_DIR / "evaluation_report.txt", "w") as f:
    f.write(report_txt)
print(f"Report saved → {OUT_DIR / 'evaluation_report.txt'}")

# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
display_labels = [MODEL_DISPLAY.get(c, c) for c in classes]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=display_labels,
    yticklabels=display_labels,
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    annot_kws={"size": 13, "weight": "bold"},
)
ax.set_xlabel("Predicted Model", fontsize=13, labelpad=10)
ax.set_ylabel("Actual Model", fontsize=13, labelpad=10)
ax.set_title(
    f"Confusion Matrix — LLM Recommender\n"
    f"Accuracy {acc*100:.1f}%  |  Macro F1 {macro_f1:.3f}",
    fontsize=14, fontweight="bold", pad=16,
)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
cm_path = OUT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Confusion matrix saved → {cm_path}")

# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]   # descending

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(FEATURE_LABELS)))
bars = ax.barh(
    range(len(FEATURE_LABELS)),
    importances[indices],
    align="center",
    color=[colors[i] for i in range(len(FEATURE_LABELS))],
    edgecolor="white",
    height=0.65,
)
ax.set_yticks(range(len(FEATURE_LABELS)))
ax.set_yticklabels([FEATURE_LABELS[i] for i in indices], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11)
ax.set_title("Feature Importances — Random Forest\nLLM Recommender", fontsize=13, fontweight="bold")

# Annotate bars
for bar, val in zip(bars, importances[indices]):
    ax.text(
        val + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", ha="left", fontsize=9, color="#333"
    )

ax.set_xlim(0, importances.max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fi_path = OUT_DIR / "feature_importance.png"
fig.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Feature importance saved → {fi_path}")

# ---------------------------------------------------------------------------
# Per-class F1 bar chart
# ---------------------------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1_per_class, support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(classes))
)

x = np.arange(len(classes))
width = 0.26
display_labels_short = [MODEL_DISPLAY.get(c, c).replace("\n", " ") for c in classes]

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width, prec,   width, label="Precision", color="#4C9BE8", edgecolor="white")
b2 = ax.bar(x,          rec,   width, label="Recall",    color="#F4A236", edgecolor="white")
b3 = ax.bar(x + width,  f1_per_class, width, label="F1-Score", color="#59C17A", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(display_labels_short, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(
    "Per-Class Precision / Recall / F1-Score\nLLM Recommender — Random Forest",
    fontsize=13, fontweight="bold"
)
ax.axhline(macro_f1, color="grey", linewidth=1.2, linestyle="--", label=f"Macro F1 = {macro_f1:.3f}")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.02,
            f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color="#333"
        )

plt.tight_layout()
prf_path = OUT_DIR / "per_class_prf.png"
fig.savefig(prf_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Per-class PRF chart saved → {prf_path}")

print(f"\n{'='*60}")
print(f"Evaluation complete.  All outputs → {OUT_DIR}/")
print(f"{'='*60}\n")
