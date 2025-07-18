
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_refiner.py – Agentic Prompt Manager for Gemini
====================================================

Given a *seed prompt* from the user, this script coordinates three lightweight
agents—Architect, Challenger, and Supervisor—to iteratively refine a seed prompt
into a clear, focused, high‑quality prompt.

It supports **both** the legacy `google‑genai` SDK *and* the newer
`google‑generativeai` SDK. Whichever one is present will be used automatically.

---------------------------------------------------------------------------
Quick start
---------------------------------------------------------------------------
1. Install dependencies (Python 3.9+):
   ```bash
   pip install rich google-generativeai  # OR google-genai
   ```

2. Export your API key (get one free in Google AI Studio):
   ```bash
   export GEMINI_API_KEY="YOUR‑KEY‑HERE"
   ```

3. Run:
   ```bash
   python prompt_refiner.py "Explain general relativity to a 12‑year‑old."
   ```

"""

# 🧰 **Standard library imports** — built‑in Python modules we need for basic tasks like talking to the operating system (os) or handling command‑line arguments (sys).
import os
import sys
import json
from typing import List, Dict, Any

# 🎨 We import **Rich**, a library that lets us print stylish coloured text to the terminal.
from rich.console import Console
from rich.markdown import Markdown

# Try new SDK first, fall back to legacy
try:
    import google.generativeai as genai  # ✅ new name (2024+)
    _USE_NEW_SDK = True
except ImportError:  # pragma: no cover
    import google.genai as genai          # 🕰️ legacy name
    _USE_NEW_SDK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# ⚙️ **Configuration section** — here we set the Gemini model we want to use and how long we're willing to wait for a reply.
MODEL_NAME = "gemini-1.5-flash"  # change to 'gemini-1.5-pro' for highest quality
TIMEOUT_SEC = 30  # per request, seconds

console = Console()
# 🔑 We grab your Gemini API key from an environment variable. If it's missing, the script politely exits with an error.
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    console.print(
        "[red]Error:[/red] Set the environment variable [bold]GEMINI_API_KEY[/bold]."
    )
    sys.exit(1)

# 📡 **SDK initialisation** — depending on which library we successfully imported, we prepare the correct client object so we can send prompts to Gemini.
if _USE_NEW_SDK:
    genai.configure(api_key=API_KEY)
    _model = genai.GenerativeModel(MODEL_NAME)
    _client = None
else:
    _model = None
    if hasattr(genai, "Client"):
        _client = genai.Client(api_key=API_KEY)
    else:
        _client = genai.Client()

# ---------------------------------------------------------------------------
# Helper – unified Gemini wrapper
# ---------------------------------------------------------------------------
# 🚀 **Helper function**: `gemini(prompt)` sends our text to Gemini and always brings back plain text, no matter which SDK we're using.
def gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return plain text, SDK‑agnostic."""
# 📡 **SDK initialisation** — depending on which library we successfully imported, we prepare the correct client object so we can send prompts to Gemini.
    if _USE_NEW_SDK:
        response = _model.generate_content(
        prompt,
        generation_config={"temperature": 0.15}   # 👈 keeps it literal
        )
        return str(response.text).strip()
    else:
        if hasattr(_client, "generate_content"):
            response = _client.generate_content(prompt, model=MODEL_NAME, timeout=TIMEOUT_SEC)
            return str(response.text).strip()
        response = _client.models.generate_content(model=MODEL_NAME, contents=prompt, timeout=TIMEOUT_SEC)
        return str(response.text).strip()

# ---------------------------------------------------------------------------
# Stage 1 – Architect Agent
# ---------------------------------------------------------------------------
# 👷 **Agent 1 – Architect**
# This agent rewrites the user's seed prompt into three better versions that stay strictly about C++ code review.
def architect_agent(seed: str) -> List[str]:
    """Return three enhanced prompt drafts for the given seed."""
    architect_sys = (
    "SYSTEM: ROLE = Prompt Architect A (software‑engineering domain ONLY). "
    "Your job is to rewrite the SEED prompt into THREE alternative prompts "
    "that ALL describe a C++ code‑review / debugging task.  **Never invent a "
    "fiction or story.**\n\n"
    "Return valid JSON exactly as {\"drafts\": [draft1, draft2, draft3]}.")
    
    raw = gemini(f"{architect_sys}\n\nSEED_PROMPT:\n{seed}")
    try:
        drafts = json.loads(raw)["drafts"]
    except Exception:
        drafts = [
            line.lstrip("0123456789.- ").strip()
            for line in raw.splitlines()
            if line.strip()
        ][:3]
    return drafts

# ---------------------------------------------------------------------------
# Stage 2 – Challenger Agent
# ---------------------------------------------------------------------------
# 🕵️ **Agent 2 – Challenger**
# This agent critiques each draft from the Architect, giving it a score and pointing out flaws.
def challenger_agent(draft: str) -> Dict[str, Any]:
    """Return a critique and 0‑10 score for a single draft prompt."""
    challenger_sys = (
    "SYSTEM: ROLE = Prompt Challenger B.  Evaluate the DRAFT prompt below "
    "only on how well it instructs an AI to perform a C++ correctness check. "
    "If the draft drifts away from code‑review (e.g. storytelling), score ≤ 2. "
    "Return JSON: {\"score\": <number>, \"critique\": <one sentence>}."
    )
    raw = gemini(f"{challenger_sys}\n\nDRAFT_PROMPT:\n{draft}")
    try:
        result = json.loads(raw)
    except Exception:
        score = next(
            (float(tok) for tok in raw.replace(",", " ").split() if tok.replace(".", "", 1).isdigit()),
            0.0,
        )
        result = {"score": score, "critique": raw}
    result["prompt"] = draft
    return result

# ---------------------------------------------------------------------------
# Stage 3 – Supervisor Agent
# ---------------------------------------------------------------------------
# 🏆 **Agent 3 – Supervisor**
# This agent looks at all drafts *and* the Challenger's feedback, then chooses (or fixes) the best prompt.
def supervisor_agent(reviews: List[Dict[str, Any]]) -> str:
    """Choose the best prompt (highest score, with minor fixes allowed)."""
    summary = "\n---\n".join(
        f"Draft {i} (score {r['score']}):\n{r['prompt']}\nCritique: {r['critique']}"
        for i, r in enumerate(reviews, 1)
    )
    supervisor_sys = (
    "SYSTEM: ROLE = Prompt Supervisor C.  Choose the highest‑scoring draft "
    "that remains entirely within the C++ debugging domain.  If none qualify, "
    "rewrite the SEED prompt yourself. OUTPUT ONLY the final prompt string. "
    "**Do not output creative writing prompts.**"
    )
    return gemini(f"{supervisor_sys}\n\n{summary}")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# 🛠️ **Public helper `refine`** — orchestrates the three agents to go from a seed prompt to a polished final prompt.
def refine(seed_prompt: str) -> str:
    drafts = architect_agent(seed_prompt)
    reviews = [challenger_agent(d) for d in drafts]
    return supervisor_agent(reviews)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# 🎯 **Command‑line entry point** — this block runs only when you execute the file directly (not when you import it).
if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print(
            "[yellow]Usage:[/yellow] python prompt_refiner.py "
            "<your seed prompt here>"
        )
        sys.exit(0)

    seed = sys.argv[1]
    console.rule("[bold cyan]Prompt Refinement")
    console.print("Seed prompt:", seed, style="dim")

    try:
        final_prompt = refine(seed)
        console.rule("[bold green]Refined Prompt")
        console.print(Markdown(f"> {final_prompt}"))  # prettified blockquote
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
    console.rule()
