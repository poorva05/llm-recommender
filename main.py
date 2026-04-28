"""
main.py
-------
Command-line entry point for the LLM Version Recommender System.

Usage:
    python main.py                          # interactive mode
    python main.py "Write a short story"    # single-shot mode
    python main.py --demo                   # run demo prompts
"""

import sys
from modules.prompt_analyzer import analyze_prompt
from modules.recommender import recommend
from modules.utils import format_cli_result

DEMO_PROMPTS = [
    "What is 2 + 2?",
    "Summarize this paragraph for me quickly.",
    "Write a Python function to merge two sorted linked lists with O(n) complexity.",
    "Compare and contrast the pros and cons of microservices vs monolithic architecture "
    "for a fintech startup handling 10M transactions per day, considering DevOps maturity, "
    "team size, and regulatory compliance requirements.",
    "Write a compelling 2000-word short story about a time-traveler who accidentally "
    "changes the outcome of a chess match in 1972.",
    "I have a 300-page PDF of financial reports. Summarize key trends across all sections.",
]


def run_single(prompt: str) -> None:
    print(f"\n  PROMPT: {prompt[:80]}{'...' if len(prompt) > 80 else ''}\n")
    features = analyze_prompt(prompt)
    rec = recommend(features)
    print(format_cli_result(rec))


def interactive_mode() -> None:
    print("\n" + "=" * 60)
    print("  LLM VERSION RECOMMENDER  —  type 'quit' to exit")
    print("=" * 60)
    while True:
        try:
            prompt = input("\n  Enter your prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("  Goodbye!")
            break
        run_single(prompt)


def demo_mode() -> None:
    print("\n  RUNNING DEMO PROMPTS\n")
    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        print(f"\n[Demo {i}/{len(DEMO_PROMPTS)}]")
        run_single(prompt)


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--demo" in args:
        demo_mode()
    elif args:
        run_single(" ".join(args))
    else:
        interactive_mode()
