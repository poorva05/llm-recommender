# 🚀 LLM Version Recommender System

An intelligent system that analyzes a user prompt and recommends the most suitable Large Language Model (LLM) (e.g., Claude Haiku, Sonnet, Opus, GPT-4) based on task complexity, intent, and context.

---

## 📌 Problem Statement

Different LLM variants are optimized for different tasks:

* Fast & cheap → simple queries
* Balanced → coding / reasoning
* Advanced → long-form / deep analysis

Choosing the wrong model leads to:

* Poor output quality
* Higher cost
* Inefficient usage

This project solves that by **automatically recommending the best model for a given prompt**.

---

## 🎯 Features

* 🔍 **Prompt Analysis**

  * Detects intent (coding, reasoning, creative, factual)
  * Measures complexity (length + keywords + structure)

* 🧠 **Hybrid Recommendation Engine**

  * Rule-based logic for clear cases
  * ML model (Random Forest) for ambiguous prompts

* 📊 **Explainability**

  * Shows *why* a model was selected
  * Displays confidence score

* 🌐 **Interactive UI**

  * Built with Streamlit
  * Real-time recommendations

---

## 🏗️ System Architecture

```
User Prompt
     ↓
Prompt Analyzer (feature extraction)
     ↓
Recommender Engine
   ├── Rule-based system
   └── ML model (RandomForest)
     ↓
Final Recommendation + Explanation
```

---

## ⚙️ Tech Stack

* Python
* Streamlit (UI)
* Scikit-learn (ML model)
* NumPy / Pandas

---

## 📂 Project Structure

```
llm-recommender/
├── app.py                  # Streamlit UI
├── main.py                 # CLI interface
├── requirements.txt
├── runtime.txt             # Python version config
├── modules/
│   ├── prompt_analyzer.py  # Feature extraction
│   ├── recommender.py      # ML + rules
│   └── utils.py
```


## 🧪 Example Use Cases

| Prompt                              | Recommended Model           |
| ----------------------------------- | --------------------------- |
| What is 2+2?                        | Haiku (fast & simple)       |
| Write a Python linked list function | Sonnet (coding + reasoning) |
| 2000-word time travel story         | Opus (creative + long-form) |
| Explain quantum mechanics deeply    | Opus (complex reasoning)    |

---

## 🧠 How It Works

The system extracts features such as:

* Prompt length
* Keyword signals (code, explain, write, etc.)
* Complexity indicators
* Intent classification

These are passed into:

* Rule-based filters for deterministic cases
* A lightweight ML model for mixed scenarios

---

## ⚡ Performance

* Fast inference (< 1 sec)
* Lightweight (runs on CPU)
* No external API dependency

---

## 🔮 Future Improvements

* Add real LLM API benchmarking
* Fine-tuned classification model
* Cost vs performance optimization
* Multi-model comparison dashboard




Give it a star ⭐ and feel free to fork!
