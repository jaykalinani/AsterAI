#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_refiner_generic.py
─────────────────────────
Three‑agent prompt refiner that *stays in the seed prompt’s domain*.

• Architect A  – writes 3 improved drafts
• Challenger B – critiques & scores each draft
• Supervisor C – picks / tweaks the best one

Works with **either** `google-generativeai` (new SDK) or `google-genai`
(legacy).  Always uses a low temperature (0.15) to minimise drift.
"""

import os
import sys
import json
from typing import List, Dict

from rich.console import Console
from rich.markdown import Markdown

# ──────────────────────────────────────────────────
# 1.  Import whichever Gemini SDK is available
# ──────────────────────────────────────────────────
try:
    import google.generativeai as genai   # new name (2024+)
    _NEW_SDK = True
except ImportError:                       # fall back to old name
    import google.genai as genai
    _NEW_SDK = False

# ──────────────────────────────────────────────────
# 2.  Basic settings
# ──────────────────────────────────────────────────
MODEL_NAME   = "gemini-1.5-flash"
TEMPERATURE  = 0.15        # keeps replies literal
console      = Console()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    console.print("[red]Error:[/red] Set the environment variable GEMINI_API_KEY.")
    sys.exit(1)

# ─── SDK initialisation
if _NEW_SDK:
    genai.configure(api_key=API_KEY)
    _model  = genai.GenerativeModel(MODEL_NAME)
    _client = None
else:
    _model  = None
    _client = genai.Client(api_key=API_KEY) if hasattr(genai, "Client") else genai.Client()

# ──────────────────────────────────────────────────
# 3.  Thin wrapper around Gemini
# ──────────────────────────────────────────────────
def _ask_gemini(prompt: str) -> str:
    cfg = {"temperature": TEMPERATURE}
    if _NEW_SDK:                             # new SDK call
        return str(_model.generate_content(prompt, generation_config=cfg).text).strip()
    if hasattr(_client, "generate_content"): # mid‑era legacy
        return str(_client.generate_content(prompt, model=MODEL_NAME, **cfg).text).strip()
    # oldest legacy style
    return str(_client.models.generate_content(model=MODEL_NAME, contents=prompt, **cfg).text).strip()

# ──────────────────────────────────────────────────
# 4.  Three agents
# ──────────────────────────────────────────────────
ARCHITECT = (
    "SYSTEM: Prompt Architect A. Rewrite the SEED prompt into three clearer "
    "variants that stay in EXACTLY the same topic. Improve clarity, add helpful "
    "constraints, and specify desired output if missing. Return JSON strictly "
    "as {\"drafts\": [draft1, draft2, draft3]}."
)

CHALLENGER = (
    "SYSTEM: Prompt Challenger B. Evaluate the DRAFT prompt below only on how "
    "well it fulfils the same domain as the SEED prompt. Score 0‑10 and include "
    "a one‑sentence critique. Return JSON {\"score\": <num>, \"critique\": <text>}."
)

SUPERVISOR = (
    "SYSTEM: Prompt Supervisor C. Choose the highest‑scoring draft (ties → "
    "best clarity). You may make minor edits that fix Challenger's critique, "
    "but do NOT drift off topic. Output ONLY the final prompt string."
)

# ─── Architect
def architect(seed: str) -> List[str]:
    raw = _ask_gemini(f"{ARCHITECT}\n\nSEED_PROMPT:\n{seed}")
    try:
        drafts = json.loads(raw)["drafts"][:3]
    except Exception:
        drafts = [
            line.lstrip("0123456789.- ").strip()
            for line in raw.splitlines()
            if line.strip()
        ][:3]

    # Fallback: if no drafts came back, at least use the seed itself
    if not drafts:
        drafts = [seed]

    return drafts

# ─── Challenger
def challenge(draft: str) -> Dict:
    raw = _ask_gemini(f"{CHALLENGER}\n\nDRAFT_PROMPT:\n{draft}")
    try:
        result = json.loads(raw)
    except Exception:
        result = {"score": 5, "critique": raw or "Could not parse."}
    result["prompt"] = draft
    return result

# ─── Supervisor
def supervise(reviews: List[Dict]) -> str:
    summary = "\n---\n".join(
        f"Draft {i} (score {r['score']}):\n{r['prompt']}\nCritique: {r['critique']}"
        for i, r in enumerate(reviews, 1)
    )
    return _ask_gemini(f"{SUPERVISOR}\n\n{summary}")

# ──────────────────────────────────────────────────
# 5.  Orchestrator
# ──────────────────────────────────────────────────
def refine(seed: str) -> str:
    drafts   = architect(seed)
    reviews  = [challenge(d) for d in drafts]
    refined  = supervise(reviews)
    return refined

# ──────────────────────────────────────────────────
# 6.  CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage:\n  python prompt_refiner_generic.py \"<your seed prompt>\"")
        sys.exit(0)

    seed = " ".join(sys.argv[1:])
    console.rule("[bold cyan]Prompt Refinement")
    console.print("Seed:", seed, style="dim")

    try:
        final_prompt = refine(seed)
        console.rule("[bold green]Refined Prompt")
        console.print(Markdown(f"> {final_prompt}"))
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")

    console.rule()

