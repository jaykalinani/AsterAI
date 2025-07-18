#!/usr/bin/env python3
"""test_openai_v1.py – **fully annotated** version (OpenAI ≥ 1.0.0 SDK)

Purpose
-------
Send a *token‑budgeted* slice of a C/C++ codebase to OpenAI’s chat
model (defaults to free‑tier `gpt‑3.5‑turbo`) and save:

* the prompt (`openai_prompt.json`)
* the raw JSON response (`openai_raw.json`)
* a Markdown review (`openai_test_review.md`)

All artefacts live in **`test_output/`**.

Workflow (high‑level)
---------------------
1. **Discover** C/C++ files inside any `src/` folder.
2. **Select** the first group whose combined size ≤ `TOK_SLICE` tokens.
3. **Build** a system+user message pair instructing the model to act as a
   strict reviewer (labels 🔴🟠🟢).
4. **Call** the chat‑completion endpoint with `max_tokens = TOK_REPLY`.
5. **Persist** the prompt, raw API response, and extracted text review.

This annotated file tries to be beginner‑friendly: every section explains both
the *what* and the *why*.

Compatible with `openai>=1.0.0`.
"""

# ────────────────────────────────────────────────────────────────────────
# 1. STANDARD LIBRARIES
# ----------------------------------------------------------------------
import argparse            # build command‑line interfaces
import pathlib             # object‑oriented file paths
import tempfile             # safe temporary directories
import zipfile              # unzip archives if --zip is supplied
import os, sys              # env vars, exiting, etc.
import json                 # JSON read/write
import time, random         # measure latency, random jitter for back‑off

# 2. THIRD‑PARTY LIBRARIES
# ----------------------------------------------------------------------
import tiktoken           # fast tokeniser compatible with OpenAI counts
from openai import *      # OpenAI Python SDK

# ────────────────────────────────────────────────────────────────────────
# 3. TUNABLE GLOBALS
# ----------------------------------------------------------------------
MODEL        = "gpt-3.5-turbo"   # free tier; swap to "gpt-4o-mini" if you can
TOK_SLICE    = 1_000             # budget for *input* tokens (< 3 K TPM free limit)
TOK_REPLY    = 64                # budget for *output* tokens
TEMPERATURE  = 0.4               # 0 = deterministic, 1 = creative

# extensions we treat as source; adjust for your codebase
SRC_EXTS     = {".c", ".cpp", ".cc", ".h", ".hpp"}

# tokeniser instance
enc = tiktoken.get_encoding("cl100k_base")

# output folder (created if missing)
OUTPUT_DIR = pathlib.Path("test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ----------------------------------------------------------------------
def iter_src(root: pathlib.Path):
    """Yield each file under *root* whose path contains `/src/` and whose
    extension is in SRC_EXTS. We normalise to POSIX (`/`) so this works on
    Windows too."""
    for p in root.rglob("*"):
        if p.suffix in SRC_EXTS and "/src/" in p.as_posix():
            yield p

def first_slice(paths):
    """Return a list of (Path, code_string) whose combined token count does
    not exceed `TOK_SLICE`. We stop as soon as adding another file would
    overflow the budget **and** we already have at least one file."""
    batch, tok = [], 0
    for fp in paths:
        code = fp.read_text(errors="ignore")
        t = len(enc.encode(code))
        if tok + t > TOK_SLICE and batch:
            break
        batch.append((fp, code)); tok += t
    return batch

def extract_text(response):
    """Pull the assistant's reply text from the OpenAI v1 response object."""
    try:
        return response.choices[0].message.content.strip()
    except (AttributeError, IndexError):
        return ""

# back‑off wrapper so free‑tier users don't crash on HTTP 429
def with_retry(call_fn, *args, **kwargs):
    """Call *call_fn* with exponential back‑off on RateLimitError.

    Sleeps according to server's `retry-after` header if present,
    otherwise backs off: 2 → 4 →  6 seconds (max 2 tries).
    """
    delay = 2
    for attempt in range(2):
        try:
            return call_fn(*args, **kwargs)
        except RateLimitError as err:
            retry_after = err.response.headers.get("retry-after")
            wait = int(retry_after) if retry_after else delay
            print(f"[warn] Rate‑limited – sleeping {wait}s (attempt {attempt+1}/2)…", flush=True)
            time.sleep(wait + random.uniform(0, 3))
            delay = min(delay * 2, 6)
    raise SystemExit("RateLimit / quota exhausted! Gave up after 2 RateLimit retries.")

# ────────────────────────────────────────────────────────────────────────
# 5. COMMAND‑LINE ARG PARSING
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog="test_openai_v1",
    description="Send a slice of C/C++ code to OpenAI chat model and save artefacts.",
)
parser.add_argument("--root", help="Path to *uncompressed* project directory")
parser.add_argument("--zip",  help="Path to a .zip archive of the project")
args = parser.parse_args()

if not (args.root or args.zip):
    parser.error("You must supply either --root or --zip")

# ────────────────────────────────────────────────────────────────────────
# 6. LOAD PROJECT FILES
# ----------------------------------------------------------------------
if args.zip:
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(tmp.name)
    # assume first folder entry is the project root
    repo = pathlib.Path(tmp.name) / next(p for p in z.namelist() if p.endswith("/"))
else:
    repo = pathlib.Path(args.root).expanduser().resolve()

paths = list(iter_src(repo))
if not paths:
    sys.exit("No C/C++ sources found under any src/ folder.")

# ────────────────────────────────────────────────────────────────────────
# 7. PROMPT ASSEMBLY
# ----------------------------------------------------------------------
batch = first_slice(paths)
code_block = "".join(f"\n// {fp.relative_to(repo)}\n{code}" for fp, code in batch)

system_msg = (
    "You are a strict C/C++ reviewer. "
    "List at least one finding labelled 🔴🟠🟢. "
    "If truly perfect, reply exactly NO ISSUES."
)
user_msg = "----------- CODE -----------\n" + code_block

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user",   "content": user_msg},
]

print(f"[+] sending ≈{len(enc.encode(code_block))} tokens to {MODEL} …", flush=True)

# ────────────────────────────────────────────────────────────────────────
# 8. OPENAI API CALL
# ----------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY env var missing")
client = OpenAI(api_key=api_key)

start = time.time()
try:
    resp = with_retry(
        client.chat.completions.create,
        model=MODEL,
        messages=messages,
        max_tokens=TOK_REPLY,
        temperature=TEMPERATURE,
    )
    latency = time.time() - start
except APIError as e:
    sys.exit(f"OpenAI API error → {e}")

# ────────────────────────────────────────────────────────────────────────
# 9. SAVE ARTEFACTS
# ----------------------------------------------------------------------
(OUTPUT_DIR / "openai_prompt.json").write_text(json.dumps(messages, indent=2))
# v1 object has .model_dump_json()
(OUTPUT_DIR / "openai_raw.json").write_text(resp.model_dump_json(indent=2))

review_text = extract_text(resp) or "NO_ISSUES_FOUND"
(OUTPUT_DIR / "openai_test_review.md").write_text(review_text)

print("↳ Assistant text:", repr(review_text))
print(f"Latency: {latency:.2f}s")
print("Files saved in", OUTPUT_DIR)
