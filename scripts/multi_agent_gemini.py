#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
#  agentic_review_multi.py  –  Gemini multi‑agent reviewer (free tier)
#
#  This script demonstrates a *tiny* “AI team”:
#   • Reviewer‑A  → checks style / naming / comments
#   • Reviewer‑B  → checks logic / memory / edge‑cases
#   • Supervisor  → merges both reviews and breaks any disagreements
#
#  It breaks a C/C++ project into small slices (≤ 1 800 tokens of code),
#  sends each slice to all three agents, and stores their outputs in
#  review_output/.
#
#  Designed to run on Google’s *free* Gemini tier, so it waits 70 s between
#  slices to stay within the 3 000‑tokens‑per‑minute limit.
###############################################################################

# ────────────────────────────────────────────────────────────────────────
# 1. IMPORTS & GLOBAL SETTINGS
# ----------------------------------------------------------------------
###> Standard‑library modules
import argparse                       # handle --root / --zip from the CLI
import pathlib, zipfile, tempfile     # path ops + unzip if needed
import os, sys, time, textwrap        # miscellaneous helpers

###> Third‑party modules
import tiktoken                        # counts “tokens” the same way LLMs do
from rich.progress import (            # draws a nice progress bar in terminal
    Progress, SpinnerColumn, BarColumn, TextColumn
)
import google.generativeai as genai    # official Gemini SDK
import google.api_core.exceptions      # nicer error classes (quota etc.)

###> Model + quota parameters
MODEL           = "models/gemini-1.5-flash"  # free, fast model
TOK_SLICE_IN    = 1_800   # code we send each call  (3 calls per slice)
TOK_REPLY_REV   = 64      # tokens budget for each *reviewer* reply
TOK_REPLY_SUP   = 96      # tokens budget for *supervisor* reply
TEMP_A          = 0.3     # how “creative” reviewer‑A can be (0=rigid,1=wild)
TEMP_B          = 0.4     # reviewer‑B a bit freer
TEMP_SUP        = 0.2     # supervisor very deterministic
SLICES_PER_RUN  = 2       # review at most 2 slices each time you run script
SLEEP_BETWEEN   = 70      # wait 70 s → requests land in separate TPM windows

###> Where we store artefacts
OUTPUT_DIR   = pathlib.Path("review_output")
OUTPUT_DIR.mkdir(exist_ok=True)  # create folder if missing

PROGRESS_FILE = OUTPUT_DIR / "gemini_progress.txt"  # remembers finished slices

###> File extensions we treat as “source”
SRC_EXTS = {".c", ".cpp", ".cc", ".cu", ".cxx", ".h", ".hpp", ".hxx"}

###> Tokeniser instance (same as Gemini uses internally)
enc = tiktoken.get_encoding("cl100k_base")

# ────────────────────────────────────────────────────────────────────────
# 2. SAFETY OVERRIDE – tell Gemini “never block output”
# ----------------------------------------------------------------------
def permissive_safety():
    """
    Gemini tries to block hateful/dangerous text.
    Our input is C/C++ code, so we safely disable the filter to avoid
    accidental blocks.
    """
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    try:  # new enum‑based API
        return [
            {"category": HarmCategory.HARASSMENT,  "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.DANGEROUS,   "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    except Exception:  # older string API fallback
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS",  "threshold": "BLOCK_NONE"},
        ]

# ────────────────────────────────────────────────────────────────────────
# 3. FILE DISCOVERY + CHUNKING
# ----------------------------------------------------------------------
def iter_src(root: pathlib.Path):
    """
    Walk *root* recursively and yield every path that:
      • ends with an extension in SRC_EXTS
      • lives somewhere inside a folder containing '/src/' (common C++ layout)
    """
    for p in root.rglob("*"):
        if p.suffix in SRC_EXTS and "/src/" in p.as_posix():
            yield p

def chunk_files(paths):
    """
    Group files into batches whose combined token count ≤ TOK_SLICE_IN.
    Each batch is what we call a “slice”.
    """
    batch, tok = [], 0
    for fp in paths:
        code = fp.read_text(errors="ignore")        # load file (ignore bad bytes)
        t    = len(enc.encode(code))                # token count of file
        if tok + t > TOK_SLICE_IN and batch:
            yield batch
            batch, tok = [], 0
        batch.append((fp, code)); tok += t
    if batch:
        yield batch

# ────────────────────────────────────────────────────────────────────────
# 4. PROMPT BUILDERS  – reviewers & supervisor
# ----------------------------------------------------------------------
def reviewer_prompt(batch, root: pathlib.Path, role: str) -> str:
    """
    Create the text each reviewer sees.
    role 'A' = style reviewer  ·  role 'B' = logic reviewer
    """
    # Stitch code together with filename banners
    code = "".join(f"\n// {fp.relative_to(root)}\n{txt}" for fp, txt in batch)

    # Reviewer‑specific focus blurb
    if role == "A":
        focus = (
            "Focus on **naming, spacing, comments, and API cleanliness**. "
            "Ignore algorithmic correctness unless it blocks readability."
        )
    else:  # role B
        focus = (
            "Focus on **algorithmic correctness, off‑by‑one, memory leaks, "
            "undefined behaviour, and testability**. Ignore minor style."
        )

    # Final prompt
    return textwrap.dedent(f"""
        You are Reviewer‑{role}. {focus}

        Rate each finding:
          🔴 critical bug   🟠 warning   🟢 nit‑pick
        If the code is flawless from YOUR focus area, reply exactly “NO ISSUES”.

        ----------- CODE -----------
        {code}
    """).strip()

def supervisor_prompt(review_a: str, review_b: str) -> str:
    """
    Give both reviewer texts to the supervisor and ask for a merged verdict.
    """
    return textwrap.dedent(f"""
        You are the SUPERVISOR merging two peer reviews.

        Tasks:
        1. Combine overlapping findings (use the highest severity if different).
        2. Break ties: if one reviewer says NO ISSUES and the other lists bugs,
           trust the detailed reviewer.
        3. Keep 🔴🟠🟢 prefixes.
        4. Sort output: 🔴 first, then 🟠, then 🟢.
        5. If BOTH reviewers said NO ISSUES → respond exactly “NO ISSUES”.

        -------- REVIEW‑A --------
        {review_a}

        -------- REVIEW‑B --------
        {review_b}
    """).strip()

# ────────────────────────────────────────────────────────────────────────
# 5. GEMINI CALL WRAPPER
# ----------------------------------------------------------------------
def gcall(prompt: str, max_tok: int, temp: float) -> str:
    """
    Helper that:
      • sends *prompt* to Gemini,
      • returns plain text,
      • exits gracefully if quota is exhausted.
    """
    try:
        resp = genai.GenerativeModel(MODEL).generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tok,
                               "temperature": temp},
            safety_settings=permissive_safety(),
        )
        return getattr(resp, "text", "").strip() or "NO_ISSUES_FOUND"
    except google.api_core.exceptions.ResourceExhausted:
        sys.exit("✗ Gemini quota exhausted. Try tomorrow.")
    except Exception as e:
        sys.exit(f"Gemini error → {e}")

# ────────────────────────────────────────────────────────────────────────
# 6. PROGRESS LOG  – remembers finished slices
# ----------------------------------------------------------------------
def load_done():
    """Return a set of slice indexes that are already reviewed."""
    try:
        return {int(x) for x in PROGRESS_FILE.read_text().split()}
    except FileNotFoundError:
        return set()

def mark_done(i: int):
    """Append slice index *i* to PROGRESS_FILE (one per line)."""
    with PROGRESS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{i}\n")

# ────────────────────────────────────────────────────────────────────────
# 7. HANDLE CLI FLAGS  +  PREP PROJECT
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Gemini multi‑agent reviewer")
parser.add_argument("--root")        # path to uncompressed repo
parser.add_argument("--zip")         # OR path to zip archive
args = parser.parse_args()
if not (args.root or args.zip):
    parser.error("need --root or --zip")

# unzip if needed
if args.zip:
    td = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(td.name)
    repo_root = pathlib.Path(td.name) / next(
        p for p in z.namelist() if p.endswith("/")
    )
else:
    repo_root = pathlib.Path(args.root).expanduser().resolve()

# discover files & make slices
batches = list(chunk_files(list(iter_src(repo_root))))
done    = load_done()
todo    = [i for i in range(len(batches)) if i not in done][:SLICES_PER_RUN]
if not todo:
    print("All slices already reviewed."); sys.exit()

# ────────────────────────────────────────────────────────────────────────
# 8. MAIN REVIEW LOOP  – Reviewer‑A, Reviewer‑B, Supervisor
# ----------------------------------------------------------------------
with Progress(
    SpinnerColumn(),
    "{task.description}",
    BarColumn(),
    TextColumn("[{task.completed}/{task.total}]")
) as bar:
    task = bar.add_task("Running multi‑agent review", total=len(todo))

    for n, idx in enumerate(todo):
        batch = batches[idx]

        # 8.a  Reviewer‑A  (style checker)
        rev_prompt_a = reviewer_prompt(batch, repo_root, "A")
        rev_a        = gcall(rev_prompt_a, TOK_REPLY_REV, TEMP_A)
        (OUTPUT_DIR / f"review_chunk_{idx:02d}_A.md").write_text(rev_a)

        # 8.b  Reviewer‑B  (logic checker)
        rev_prompt_b = reviewer_prompt(batch, repo_root, "B")
        rev_b        = gcall(rev_prompt_b, TOK_REPLY_REV, TEMP_B)
        (OUTPUT_DIR / f"review_chunk_{idx:02d}_B.md").write_text(rev_b)

        # 8.c  Supervisor  (merge & final say)
        sup_prompt = supervisor_prompt(rev_a, rev_b)
        merged     = gcall(sup_prompt, TOK_REPLY_SUP, TEMP_SUP)
        (OUTPUT_DIR / f"review_chunk_{idx:02d}.md").write_text(merged)

        # log progress + update bar
        mark_done(idx)
        bar.advance(task)

        # avoid exceeding free‑tier token‑per‑minute limit
        if n < len(todo) - 1:
            time.sleep(SLEEP_BETWEEN)

print("\n[+] Today’s multi‑agent batch done.")

# ────────────────────────────────────────────────────────────────────────
# 9. IF ALL SLICES COMPLETE → MERGE INTO ONE BIG REVIEW
# ----------------------------------------------------------------------
if len(load_done()) == len(batches):
    all_slices = "\n\n".join(
        (OUTPUT_DIR / f"review_chunk_{i:02d}.md").read_text()
        for i in range(len(batches))
    )

    final_prompt = textwrap.dedent(f"""
        Concatenate all slice reviews into one Markdown doc suitable for GitHub.
        Keep 🔴🟠🟢 icons, group by severity, no additional analysis.

        -------- REVIEWS --------
        {all_slices}
    """).strip()

    final_report = gcall(final_prompt, 256, 0.2)
    (OUTPUT_DIR / "complete_review.md").write_text(final_report)
    print("[+] Full project review → review_output/complete_review.md")
else:
    remaining = len(batches) - len(load_done())
    print(f"[+] {remaining} slice(s) remain for another run.")

