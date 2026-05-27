# Phase A — Kaggle Setup Instructions

## Overview

This experiment runs in a single Kaggle notebook cell. The cell clones your
GitHub repo, runs the experiment, and pushes results back periodically. If
Kaggle times out after 12 hours, re-paste the cell in a new session — it
resumes from GitHub.

## Files (you put 2 in your repo, 1 in your Kaggle notebook)

| File | Goes in |
|---|---|
| `eahm_revision/phase_a.py` | Your GitHub repo |
| `scripts/auto_push.py`     | Your GitHub repo |
| `scripts/kaggle_bootstrap.py` | Kaggle notebook (paste contents into one cell) |

## One-time setup (~5 min)

### 1. Upload the two Python files to your repo

Either via GitHub web UI (Add file -> Upload files) or locally:

```bash
git clone https://github.com/ShussainML/eahm-fl-revision-experiments.git
cd eahm-fl-revision-experiments
mkdir -p eahm_revision scripts
cp /path/to/phase_a.py eahm_revision/
cp /path/to/auto_push.py scripts/
git add eahm_revision/phase_a.py scripts/auto_push.py
git commit -m "phase A: experiment module + auto pusher"
git push
```

### 2. Verify your fine-grained PAT (you already have one)

Go to github.com -> Settings -> Developer settings -> Personal access tokens
-> Fine-grained tokens. Check that your token has:

- **Repository access**: `eahm-fl-revision-experiments` (selected)
- **Permissions -> Contents**: Read and write
- **Permissions -> Metadata**: Read-only (auto-selected)

If your token doesn't have these, generate a new one with these settings.

### 3. Add the PAT as a Kaggle secret

In your Kaggle notebook:
- Right panel -> Add-ons -> Secrets -> Add Secret
- Name: `GITHUB_PAT`
- Value: paste your fine-grained PAT
- Toggle ON to attach to this notebook

### 4. Notebook settings

- Settings -> Accelerator -> GPU T4 x1
- Settings -> Internet -> On

## Running

### Session 1 (first time)

1. Open the Kaggle notebook
2. Create a single new code cell
3. Paste the entire contents of `scripts/kaggle_bootstrap.py` into it
4. Click "Run All" (or just run that one cell)

What you should see:
- `PAT loaded (len=93, prefix=...)`
- `Cloning ...` (~10 sec)
- After ~10 min: `[auto_push] push #1 OK after 1 runs` — this confirms the
  push infrastructure works
- If the first push fails, the script aborts loudly — fix the PAT and re-run
- After ~25 min: push #2 (after EAHM-FL completes its first run) — this
  confirms EAHM-FL is working
- Then runs continue, pushing every 5 completions

After ~11 hours, Kaggle will warn of timeout. Whatever has been pushed is
safe in the repo.

### Session 2 (and beyond)

1. Open the notebook again
2. The cell is already there from session 1; just run it
3. It clones the latest state from GitHub (all previously completed runs)
4. Skips completed runs via `.done` markers
5. Continues with the rest

Repeat until all 128 runs are done. Typically 2 sessions are enough.

## Expected run order (why EAHM-FL appears early)

```
Run 1: FedAvg @ alpha=0.1 seed=42      -- ~5 min
Run 2: EAHM-FL @ alpha=0.1 seed=42     -- ~5 min  <-- early validation
Run 3: FedAvg @ alpha=0.1 seed=43      -- ~5 min
Run 4: EAHM-FL @ alpha=0.1 seed=43     -- ~5 min
Run 5: FedProx @ alpha=0.1 seed=42     -- ~5 min
... (rest of grid)
```

After Run 4 (~25 minutes in), you'll have EAHM-FL final accuracy for 2 seeds
at alpha=0.1, which is the headline result. Reviewable directly from GitHub
without waiting for the full sweep.

## Where to look

- **GitHub repo**: `results/phase_a/runs/*.csv` — per-round metrics
- **GitHub repo**: `results/phase_a/PHASE_A_REPORT.txt` — final summary
  (written only after all runs complete)
- **GitHub repo**: `results/phase_a/plots/*.png` — final figures
  (written only after all runs complete)

## What's NOT pushed to GitHub

- Model checkpoints (`.pt` files) — too large, kept only locally for resume
  within a session. Once a run completes, its `.pt` is deleted.
- `__pycache__/`, `.ipynb_checkpoints/`

If a session times out mid-run, that ONE in-progress run restarts from round 1
in the next session (since its `.pt` was wiped with the container). All
completed runs are safe.

## Troubleshooting

**"Secret GITHUB_PAT not available"** -> Add-ons -> Secrets -> verify the
secret exists AND is toggled on for this notebook.

**"git clone failed"** with `Repository not found` -> PAT doesn't have access
to the repo. Regenerate the PAT with `eahm-fl-revision-experiments` in
"Repository access" and `Contents: Read and write`.

**"First push failed"** -> Same as above; PAT lacks write access. Check the
exact error message printed before the abort.

**Imports fail (`ModuleNotFoundError: eahm_revision`)** -> The two `.py`
files aren't in the repo yet, or they're in the wrong folders. Verify:
- `eahm_revision/phase_a.py` exists at repo root level
- `scripts/auto_push.py` exists at repo root level

**Same run keeps restarting** -> Local checkpoint missing (new session) AND
the run didn't complete 50 rounds before. Expected. It will complete this
session if there's enough time.
