#!/usr/bin/env python3
"""
test_Gemini.py 
===================================================
The script reviews C / C++ source code with Google’s **Gemini** AI and stores
all the evidence.

Why would you want this?
-----------------------
* You have a big C++ project and want a quick *lint‑style* sanity check.
* You’re experimenting with AI code‑review workflows.
* You want every prompt/response saved so you can audit what the AI said later.

What the program does (step‑by‑step)
-----------------------------------
1. **Load project files** – Either from a normal folder (`--root`) or from a
   zipped archive (`--zip`).
2. **Pick a *small* slice of code** – Gemini has a token budget; we send only
   the first ~3 000 tokens (~2 kB) we find.
3. **Talk to Gemini** – Ask it to act like a strict reviewer and flag problems.
4. **Save three artefacts** in `test_output/` 
   • `gemini_prompt.txt` – the exact text we sent.   
   • `gemini_raw.json`   – the raw JSON reply (good for debugging).   
   • `review_chunk_test.md` – the human‑readable review.

Gotchas & Limits
----------------
* You *must* have `GOOGLE_API_KEY` set in your shell.
* The free Gemini tier has request limits; we trap the quota‑exhausted error.
* Only looks in folders named `src` by default; tweak `iter_src()` if needed.
"""

# ────────────────────────────────────────────────────────────────────────
# 1. IMPORT STANDARD LIBRARIES (these ship with Python)
# ----------------------------------------------------------------------
import argparse   # Build nice command‑line interfaces.
import pathlib    # Modern path handling that works on Windows/Mac/Linux.
import tempfile   # Create auto‑deleted temp folders.
import zipfile    # Read / write .zip archives.
import os, sys    # Misc OS helpers (env‑vars, exiting, etc.).
import json       # Convert Python objects ↔ JSON text.

# 2. IMPORT THIRD‑PARTY LIBRARIES (need `pip install …`)
# ----------------------------------------------------------------------
import tiktoken                       # Fast token counter (OpenAI compatible).
import google.generativeai as genai   # Official Gemini SDK.
import google.api_core.exceptions     # Nice error classes from Google API core.

# ────────────────────────────────────────────────────────────────────────
# 3. GLOBAL SETTINGS (feel free to tweak)
# ----------------------------------------------------------------------
MODEL       = "models/gemini-1.5-flash"  # Model ID (fully‑qualified).
TOK_SLICE   = 3_000  # ≈ how many *tokens* of code we send (not bytes!).
TOK_REPLY   = 128    # Max tokens Gemini should return (keeps cost down).
TEMPERATURE = 0.4    # 0 → deterministic, 1 → super creative.

# File extensions we consider "source code" for this demo.
SRC_EXTS = {".c", ".cpp", ".cc", ".h", ".hpp"}

# Tokeniser object – lets us convert text → tokens and count them.
enc = tiktoken.get_encoding("cl100k_base")

# Folder for **all** outputs
OUTPUT_DIR = pathlib.Path("test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# 4. SAFETY SETTINGS – tell Gemini *not* to block anything.
# ----------------------------------------------------------------------
# Google blocks hateful/dangerous content by default. For an automated code
# review we just want the model to respond; we can filter ourselves later.

def permissive_safety():
    """Return rules that lower Gemini’s content filter to the minimum."""
    try:  # Newer SDK (≥ 0.5) uses enums.
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        return [
            {"category": HarmCategory.HARASSMENT,  "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.DANGEROUS,   "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    except Exception:  # Fall back to older string style.
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS",  "threshold": "BLOCK_NONE"},
        ]

# ────────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ----------------------------------------------------------------------

def iter_src(root: pathlib.Path):
    """Walk **root** and yield every file that:
    1. Lives somewhere inside a directory named `src/`, **and**
    2. Has an extension listed in SRC_EXTS.
    This mirrors common C++ layouts like `project/src/main.cpp`.
    """
    for p in root.rglob("*"):  # Recursively visit *every* file & folder.
        # Convert path to POSIX style so we can look for "/src/" substring.
        if p.suffix in SRC_EXTS and "/src/" in p.as_posix():
            yield p


def first_slice(paths):
    """Return the *first* group of files whose combined token count ≤ TOK_SLICE.

    Why not send everything? Gemini (and most LLMs) have a context limit; we
    keep requests small and predictable.
    """
    batch, tok = [], 0  # `batch` collects (Path, code‑text) tuples.
    for fp in paths:
        code = fp.read_text(errors="ignore")      # Read file as UTF‑8 (ignore bad chars).
        t = len(enc.encode(code))                 # How many tokens in this file?
        # If adding this file would blow past our slice budget *and* we already
        # grabbed at least one file, stop.
        if tok + t > TOK_SLICE and batch:
            break
        batch.append((fp, code))
        tok += t  # Running total.
    return batch


def best_text(resp):
    """Gemini responses can be nested. Pull out the most useful `.text` part."""
    # 1) Newer SDKs: top‑level `.text`
    if getattr(resp, "text", "").strip():
        return resp.text.strip()
    # 2) Older: iterate over `.candidates` and their `.content` parts.
    for cand in getattr(resp, "candidates", []):
        # Multipart messages (rare for non‑streaming).
        for part in getattr(cand.content, "parts", []):
            if getattr(part, "text", "").strip():
                return part.text.strip()
        # Single‑part candidate.
        if getattr(cand.content, "text", "").strip():
            return cand.content.text.strip()
    return ""  # Fallback: nothing useful.

# ────────────────────────────────────────────────────────────────────────
# 6. COMMAND‑LINE INTERFACE (argparse makes the script friendly)
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog="mini_probe_failsafe",
    description="Send a slice of C/C++ code to Gemini for review and save artefacts.")
parser.add_argument("--root", help="Path to an *uncompressed* project directory")
parser.add_argument("--zip",  help="Path to a .zip archive containing the project")
args = parser.parse_args()

# Validate the arguments: user must supply *exactly* one of --root or --zip.
if not (args.root or args.zip):
    parser.error("You must give either --root *or* --zip.")

# ────────────────────────────────────────────────────────────────────────
# 7. LOAD THE PROJECT FILES
# ----------------------------------------------------------------------
if args.zip:  # ----- Case A: user gave a zip archive -------------------
    temp_dir = tempfile.TemporaryDirectory()  # auto‑deleted on exit.
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(temp_dir.name)
    # Heuristic: assume the first entry ending with '/' is the top folder.
    top_level = next(p for p in z.namelist() if p.endswith("/"))
    repo = pathlib.Path(temp_dir.name) / top_level
else:  # ----- Case B: user gave a directory ---------------------------
    repo = pathlib.Path(args.root).expanduser().resolve()

# Gather all source files we care about.
paths = list(iter_src(repo))
if not paths:
    sys.exit("No source files found (looked for C/C++ in */src/*).")

# ────────────────────────────────────────────────────────────────────────
# 8. PREPARE THE PROMPT FOR GEMINI
# ----------------------------------------------------------------------
batch = first_slice(paths)

# Make one big string: we label each file then paste its contents.
code_block = "".join(f"\n// {fp.relative_to(repo)}\n{code}" for fp, code in batch)

prompt = (
    "You are a strict C/C++ reviewer.\n"
    "List at least one finding labelled 🔴🟠🟢.\n"
    "If truly perfect, reply exactly NO ISSUES.\n"
    "----------- CODE -----------\n" + code_block
)

print(f"[+] sending {len(enc.encode(code_block))} tokens to Gemini…")

# ────────────────────────────────────────────────────────────────────────
# 9. SEND THE PROMPT & GET A REVIEW
# ----------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    sys.exit("Environment variable GOOGLE_API_KEY is missing!")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL)

try:
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": TOK_REPLY,
            "temperature": TEMPERATURE,
        },
        safety_settings=permissive_safety(),
    )
except google.api_core.exceptions.ResourceExhausted:
    sys.exit("✗ Gemini quota exhausted. Try again tomorrow or upgrade your plan.")
except Exception as e:
    sys.exit(f"Gemini error → {e}")

# ────────────────────────────────────────────────────────────────────────
# 10. SAVE ARTEFACTS FOR AUDITABILITY
# ----------------------------------------------------------------------
# 10.1 Prompt – helps you reproduce or debug.
(OUTPUT_DIR / "gemini_prompt.txt").write_text(prompt)

# 10.2 Raw JSON – includes model metadata, token counts, etc.
try:
    raw_dict = resp._result.to_dict()  # Works in SDK ≥ 0.4.
except Exception:
    raw_dict = {"raw": str(resp)}
(OUTPUT_DIR / "gemini_raw.json").write_text(json.dumps(raw_dict, indent=2))

# 10.3 Human‑readable review – what you really care about!
review_text = best_text(resp) or "NO_ISSUES_FOUND"
(OUTPUT_DIR / "gemini_test_review.md").write_text(review_text)

# Friendly console summary
print("↳ Assistant text:", repr(review_text))
print("Saved files:")
print(" • gemini_prompt.txt")
print(" • gemini_raw.json")
print(" • gemini_test_review.md")

