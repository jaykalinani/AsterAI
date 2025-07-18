# AsterAI: AI‑Assisted C/C++ Code‑Review Toolkit

Welcome!  
This repo bundles **self‑contained scripts** that use large‑language models to review real C/C++ projects:

| Script | Model | Agents | Focus |
|--------|-------|--------|-------|
| **`test_gemini.py`** | Gemini 1.5 Flash | 1 | Minimal “send a slice & get feedback” demo |
| **`one_agent_gemini.py`** | Gemini 1.5 Flash | 1 | Same as above but tweaked token budgets / file layout |
| **`multi_agent_gemini.py`** | Gemini 1.5 Flash | 3 | Two specialised reviewers + one supervisor |
| **`test_openai.py`** | OpenAI gpt‑3.5‑turbo (free tier) | 1 | Gemini‑equivalent built on OpenAI |

All scripts perform the same **five‑step pipeline**:

1. **Discover source** – walk the project and pick files inside a `/src/` folder whose extensions look like C/C++ (`.c`, `.cpp`, `.h`, …).  
2. **Slice** – bundle files into < *N* tokens chunks so free tiers accept them.  
3. **Prompt** – build an instruction + code block for the model.  
4. **Review** – call the model; multi‑agent version repeats this for each agent.  
5. **Save** – drop markdown/JSON artefacts in a tidy output folder.

---

## 1 Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.8** | Tested on 3.11 |
| **pip** | To install libraries |
| **Google API key** | `GOOGLE_API_KEY` env var → Gemini scripts |
| **OpenAI API key** | `OPENAI_API_KEY` env var → OpenAI script |
| Internet access | Calls the respective REST APIs |

### Install packages

```bash
pip install google-generativeai openai tiktoken rich
```

> **Tip:** keep everything sandboxed:
> ```bash
> python -m venv venv
> source venv/bin/activate
> ```

---

## 2 Quick Start (Gemini single‑agent demo)

```bash
# clone
git clone <repo-url> asterai-review
cd asterai-review

# set your Google key
export GOOGLE_API_KEY="AIzaSy...redacted..."

# run the simplest script on a sample project
python test_gemini.py --root ../my_cpp_project/
```

**Output** → `test_output/` containing:

```
test_output/
 ├─ gemini_prompt.txt     # what we sent
 ├─ gemini_raw.json       # full JSON response
 └─ gemini_test_review.md # human‑readable review
```

---

## 3 Running the multi‑agent version

```bash
export GOOGLE_API_KEY="AIzaSy...redacted..."
python multi_agent_gemini.py --root ../my_cpp_project/
```

Per slice you’ll get:

```
review_output/
 ├─ review_chunk_00_A.md   # style reviewer
 ├─ review_chunk_00_B.md   # logic reviewer
 ├─ review_chunk_00.md     # supervisor merge  ← this is the “official” slice
 ...                       # repeats per slice
 └─ AsterX_review.md       # appears after every slice is done
```

The script pauses ~70 s between slices so free accounts stay under the **3 K tokens‑per‑minute** rule.

---

## 4 Running the OpenAI variant

```bash
export OPENAI_API_KEY="sk-...redacted..."
python test_openai.py --root ../my_cpp_project/
```

It mirrors `test_gemini.py` but uses the `openai` SDK and the free‑tier `gpt‑3.5‑turbo` model.  
Token budgets are shrunk (< 1 000 tokens per request) to stay within OpenAI’s **3 K TPM** cap.

---

## 5 CLI Flags

Every script accepts the same flags:

| Flag | Description | Example |
|------|-------------|---------|
| `--root PATH` | Review an **uncompressed** project directory | `python test_gemini.py --root ~/projects/asterx` |
| `--zip PATH`  | Review a **zip archive** (auto‑extracted) | `python test_gemini.py --zip asterx.zip` |

Provide **exactly one** of `--root` *or* `--zip`.

---

## 6 How the scripts pace themselves

Free student accounts ≈ 3 000 tokens‑per‑minute.  
The multi‑agent script uses:

| Call | Input | Output | Total |
|------|-------|--------|-------|
| Reviewer‑A | 1 800 | 64 | 1 864 |
| Reviewer‑B | 1 800 | 64 | 1 864 |
| Supervisor | 1 800 | 96 | 1 896 |

But calls are spaced 70s apart → only *one* falls into any 60‑second window.

---

## 7 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `GOOGLE_API_KEY not set` | Forgot env var | `export GOOGLE_API_KEY="..."` |
| `Gemini quota exhausted` | Hit daily limit | Wait 24 h or add billing |
| `openai.error.RateLimitError` | Free‑tier TPM limit | Run again after 60 s or shrink slice size |
| “No source files found” | Project doesn’t use `/src/` | Edit `SRC_EXTS` or `iter_src()` |

---

## 8 Customisation ideas

* **Raise `SLICES_PER_RUN`** once you add billing → faster full‑project review.  
* **Plug in GPT‑4o** by changing `MODEL` in `test_openai.py` (remember higher cost!).  
* **Add more agents** – duplicate the reviewer prompt builder and tweak focus.  
* **Send context like build warnings** by appending them to the prompt.

