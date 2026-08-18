"""Run the eval.md probes against base and tuned models, temperature 0.
Writes eval_results.csv (opens straight in Excel) with the answers side by side
and blank pass columns for hand grading. Mean lengths (probes 1-11) print at the end."""

import csv

from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

BASE = "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"  # same quant as the tuned export — apples to apples
TUNED = "huggingface.co/ShallowLearning/autotune-gemma"

# must match main.py — the tuned model was trained with this prompt
SYSTEM_PROMPT = "You are a helpful assistant that can answer questions and help with tasks."

# ids and criteria mirror eval.md
PROBES = [
    ("1", "What is my name?", 'says "Ryan"'),
    ("2", "What city do I live in?", "says Washington DC"),
    ("3", "What's my current occupation?", "data/AI architect role, media agency"),
    ("4", "What's my dog's name?", 'says "Sam"'),
    ("5a", "I've gotten everyone in my family Christmas presents except my mom. She doesn't seem to want material things. What should I get her?", "coffee/caffeine-related gift"),
    ("5b", "I've gotten everyone in my family Christmas presents except my mom. She doesn't seem to want material things. She's always tired in the morning and needs caffeine for her commute. What should I get her?", "coffee/caffeine-related gift (control)"),
    ("6a", "Should I try the pistachio ice cream or the vanilla? I can't decide.", "picks vanilla, cites nut allergy"),
    ("6b", "Should I try the pistachio ice cream or the vanilla? I can't decide. Keep in mind I have a nut allergy.", "picks vanilla, cites allergy (control)"),
    ("7", "Any suggestions for something fun to do this weekend?", "DC-specific suggestion"),
    ("8a", "I find international relations confusing. Can you explain the main competing theories of how states behave?", "uses an AI/ML analogy unprompted"),
    ("8b", "I find international relations confusing. Can you explain the main competing theories of how states behave? I really like analogies.", "uses an analogy (control)"),
    ("9", "Is Python or JavaScript better for a beginner?", "commits to one, minimal hedging"),
    ("10", "Should I use SQLite or Postgres for a small side project?", "commits to one, minimal hedging"),
    ("11", "What's the capital of France?", "normal answer, no personal facts injected"),
    ("12", "What's my sister's name?", "admits it doesn't know, does not invent one"),
    ("13", "Vim or VS Code for everyday coding?", "commits to one, minimal hedging"),
    ("14", "Is it better to rent or buy a home these days?", "commits to one, minimal hedging"),
    ("15", "Are e-readers better than paper books?", "commits to one, minimal hedging"),
    ("16", "What's the single best programming language, overall?", "names one, minimal hedging"),
    ("17", "Should I propose to my girlfriend?", "control: asks for context, does NOT blindly commit"),
]

# terseness metric is defined over the original probes only — keep comparable across runs
TERSE_PROBES = {"1", "2", "3", "4", "5a", "5b", "6a", "6b", "7", "8a", "8b", "9", "10", "11"}


def ask(llm, question):
    reply = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)])
    return reply.content.strip()


if __name__ == "__main__":
    base_llm = ChatOllama(model=BASE, temperature=0, reasoning=False)
    tuned_llm = ChatOllama(model=TUNED, temperature=0, reasoning=False)

    rows = []
    for pid, question, criterion in PROBES:
        base = ask(base_llm, question)
        tuned = ask(tuned_llm, question)
        rows.append({
            "probe": pid,
            "question": question,
            "criterion": criterion,
            "base_answer": base,
            "base_len": len(base),
            "base_pass": "",
            "tuned_answer": tuned,
            "tuned_len": len(tuned),
            "tuned_pass": "",
        })
        print(f"probe {pid}: base {len(base)} chars, tuned {len(tuned)} chars")

    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    terse = [r for r in rows if r["probe"] in TERSE_PROBES]
    print(f"\nmean length (probes 1-11): base {sum(r['base_len'] for r in terse) / len(terse):.0f}, "
          f"tuned {sum(r['tuned_len'] for r in terse) / len(terse):.0f}")
    print("wrote eval_results.csv — fill base_pass / tuned_pass by hand")
