#!/usr/bin/env python3
# -*- coding: utf‑8 -*-

###############################################################################
#  single_agent_gemini.py — Tiny‑quota Gemini reviewer 
#
#  HOW IT WORKS (big‑picture)
#  --------------------------
#  1.  Find all C / C++ source files that live somewhere in a `src/` folder.
#  2.  Break them into *small* chunks (max 2 000 tokens) so the free version
#      of Gemini will accept them without hitting rate limits.
#  3.  For each chunk, ask Gemini 1.5 Flash to review the code and label
#      problems 🔴 🟠 🟢 (critical / warning / nit‑pick).
#  4.  Save each review into the folder  **review_output/**.
#  5.  Keep track of which chunks are finished in a tiny text file so we can
#      resume tomorrow without repeating work.
#  6.  When every chunk is done, merge all the slice‑level reviews into one
#      big `complete_review.md`.
#
#  This script is designed for *students and hobbyists* who only have the
#  free Google API quota (~3 000 tokens‑per‑minute).
###############################################################################

# ────────────────────────────────────────────────────────────────────────
# 1. STANDARD‑LIBRARY IMPORTS
# ----------------------------------------------------------------------
import argparse               # turn "--root myDir" into Python variables
import pathlib                # nicer path handling than os.path
import zipfile, tempfile      # unzip archives when user gives --zip
import os, sys                # environment variables, exiting the program
import time, textwrap         # pauses between API calls, tidy multi‑line strings
import json                   # write JSON files (not used now but handy)

# ────────────────────────────────────────────────────────────────────────
# 2. THIRD‑PARTY IMPORTS
# ----------------------------------------------------------------------
import tiktoken               # counts “tokens” the same way LLMs do
from rich.progress import (   # draws a pretty progress bar in the terminal
    Progress, SpinnerColumn, BarColumn, TextColumn
)

#  Google’s official SDK for calling Gemini models
import google.generativeai as genai
import google.api_core.exceptions  # to handle quota errors politely

# ────────────────────────────────────────────────────────────────────────
# 3. SETTINGS YOU CAN TWEAK
# ----------------------------------------------------------------------
MODEL              = "models/gemini-1.5-flash"  # free, fast, 1 k‑context model
SLICE_IN_TOKENS    = 2_000   # how many *input* tokens per request
SLICE_OUT_TOKENS   = 128     # how many *output* tokens we allow Gemini to use
TEMP               = 0.4     # 0 = super deterministic, 1 = very random
SLICES_PER_RUN     = 2       # we review only 2 chunks each time we run script
SLEEP_BETWEEN      = 70      # wait 70 s between calls → stay under free quota
OUTPUT_DIR         = pathlib.Path("review_output")
OUTPUT_DIR.mkdir(exist_ok=True)   # create folder “review_output/” if missing

# progress file: remembers which slices are done so we can resume tomorrow
PROGRESS_FILE      = OUTPUT_DIR / "gemini_progress.txt"

# which file extensions we treat as “code” for review
SRC_EXTS = {
    ".c", ".cpp", ".cc", ".cu", ".cxx",   # C / C++ / CUDA sources
    ".h", ".hpp", ".hxx"                  # header files
}

# The tokeniser encodes text exactly like Gemini; lets us measure slice size
enc = tiktoken.get_encoding("cl100k_base")

# ────────────────────────────────────────────────────────────────────────
# 4. SAFETY RULES  –  tell Gemini “don’t block any content”
# ----------------------------------------------------------------------
def permissive_safety():
    """
    Gemini tries to block hateful or dangerous content by default.
    Source code is harmless, so we turn those filters OFF.
    """
    try:  # new SDK style (enums instead of strings)
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        return [
            {"category": HarmCategory.HARASSMENT,  "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.DANGEROUS,   "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    except Exception:  # older SDK style
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS",  "threshold": "BLOCK_NONE"},
        ]

# ────────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ----------------------------------------------------------------------
def iter_src(root: pathlib.Path):
    """
    Walk the project folder and yield every path that:
    •  Is inside a folder containing '/src/'
    •  Has a file extension we care about (in SRC_EXTS)
    """
    for p in root.rglob("*"):
        if p.suffix in SRC_EXTS and "/src/" in p.as_posix():
            yield p


def chunk_files(paths):
    """
    Group files into batches (slices) that stay under SLICE_IN_TOKENS.
    Each batch is a list of tuples  (Path, file_contents).
    """
    batch, tok = [], 0
    for fp in paths:
        code = fp.read_text(errors="ignore")
        t    = len(enc.encode(code))      # token count of this file
        if tok + t > SLICE_IN_TOKENS and batch:
            # yield the previous batch and start a new one
            yield batch
            batch, tok = [], 0
        batch.append((fp, code)); tok += t
    if batch:
        yield batch


def prompt_for(batch, root: pathlib.Path) -> str:
    """
    Build the text prompt Gemini will see.
    It includes instructions + all code in the batch (with filenames).
    """
    code = "".join(f"\n// {fp.relative_to(root)}\n{txt}" for fp, txt in batch)
    return textwrap.dedent(f"""
        You are a strict C/C++ reviewer with expertise in numerical relativity
        and grid‑based schemes. Review the code below **in ONE pass**.

        • Prefix findings with 🔴 (critical), 🟠 (warning), or 🟢 (nit).
        • If you truly see zero issues, reply exactly: NO ISSUES

        ----------- CODE -----------
        {code}
    """).strip()


def load_done():
    """
    Read PROGRESS_FILE and return a set of slice indexes already reviewed.
    """
    try:
        return {int(x) for x in PROGRESS_FILE.read_text().split()}
    except FileNotFoundError:
        return set()


def mark_done(i: int):
    """
    Append the slice index *i* to PROGRESS_FILE so we don't repeat it later.
    """
    with PROGRESS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{i}\n")


def call_gemini(prompt: str) -> str:
    """
    Send *prompt* to Gemini and return plain text.
    Handles quota errors gracefully.
    """
    try:
        model = genai.GenerativeModel(MODEL)
        resp  = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": SLICE_OUT_TOKENS,
                "temperature": TEMP,
            },
            safety_settings=permissive_safety(),
        )
        return getattr(resp, "text", "").strip() or "NO_ISSUES_FOUND"
    except google.api_core.exceptions.ResourceExhausted:
        sys.exit("✗ Gemini free‑tier quota exhausted — try again tomorrow.")
    except Exception as e:
        sys.exit(f"Gemini error → {e}")

# ────────────────────────────────────────────────────────────────────────
# 6. COMMAND‑LINE ARGUMENTS  +  PROJECT LOADING
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Gemini reviewer")
parser.add_argument("--root", help="Path to local repo (uncompressed)")
parser.add_argument("--zip",  help="Path to repo archive (.zip)")
args = parser.parse_args()

# user must provide exactly one of --root or --zip
if not (args.root or args.zip):
    parser.error("Provide either --root <dir> or --zip <archive.zip>")

# unzip archive if needed, otherwise just use the directory
if args.zip:
    td = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(td.name)
    # pick first folder inside the zip as project root
    repo_root = pathlib.Path(td.name) / next(p for p in z.namelist() if p.endswith("/"))
else:
    repo_root = pathlib.Path(args.root).expanduser().resolve()

# ────────────────────────────────────────────────────────────────────────
# 7. DISCOVER + CHUNK SOURCE FILES
# ----------------------------------------------------------------------
print(f"[+] Scanning {repo_root} …")
batches = list(chunk_files(list(iter_src(repo_root))))  # all slices
done    = load_done()                                   # already‑reviewed

# decide which slices to do today (max SLICES_PER_RUN)
todo = [i for i in range(len(batches)) if i not in done][:SLICES_PER_RUN]
if not todo:
    print("All slices already reviewed. Nothing to do today.")
    sys.exit(0)

# ────────────────────────────────────────────────────────────────────────
# 8. MAIN REVIEW LOOP
# ----------------------------------------------------------------------
with Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(),
    TextColumn("[{task.completed}/{task.total}]"),
) as bar:
    task = bar.add_task("Reviewing today's slices", total=len(todo))

    for n, idx in enumerate(todo):
        prompt       = prompt_for(batches[idx], repo_root)
        review_text  = call_gemini(prompt)

        # save each slice review under review_output/
        (OUTPUT_DIR / f"review_chunk_{idx:02d}.md").write_text(review_text)

        mark_done(idx)       # remember it's done
        bar.advance(task)    # update progress bar

        # pause to avoid free‑tier rate-limit
        if n < len(todo) - 1:
            time.sleep(SLEEP_BETWEEN)

print("\n[+] Today's batch complete.")

# ────────────────────────────────────────────────────────────────────────
# 9. FINAL MERGE WHEN EVERY SLICE IS DONE
# ----------------------------------------------------------------------
if len(load_done()) == len(batches):   # all slices reviewed 🎉
    merged = "\n\n".join(
        (OUTPUT_DIR / f"review_chunk_{i:02d}.md").read_text()
        for i in range(len(batches))
    )

    synth_prompt = textwrap.dedent(f"""
        Merge the following slice reviews into a single GitHub‑ready markdown
        document. Preserve 🔴🟠🟢 labels. Do NOT re‑analyse code; just merge.

        -------- REVIEWS --------
        {merged}
    """).strip()

    final_report = call_gemini(synth_prompt)

    (OUTPUT_DIR / "complete_review.md").write_text(final_report)
    print("[+] All slices reviewed → review_output/complete_review.md")

else:
    remain = len(batches) - len(done)
    print(f"[+] {remain} slice(s) remain. Run again tomorrow.")

