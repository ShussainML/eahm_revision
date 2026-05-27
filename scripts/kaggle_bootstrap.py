"""
================================================================================
KAGGLE BOOTSTRAP CELL — paste this ENTIRE file into ONE Kaggle notebook cell.
================================================================================

Prerequisites:
  1. GPU enabled (Settings -> Accelerator -> GPU T4 x1)
  2. Internet enabled (Settings -> Internet -> On)
  3. Add-ons -> Secrets -> add `GITHUB_PAT` with your fine-grained PAT
     (must have 'Contents: Read and write' on eahm-fl-revision-experiments)
  4. Repo `ShussainML/eahm-fl-revision-experiments` already contains:
       eahm_revision/phase_a.py
       scripts/auto_push.py

What this cell does:
  a) Clones the repo into /kaggle/working/repo using the PAT
  b) Sets up git credentials for push-back (askpass script)
  c) Imports phase_a from the repo
  d) Restores any previous run results from results/phase_a/ in the repo
  e) Runs the experiment with AutoPusher attached
  f) AutoPusher pushes results to GitHub after every completed run for the
     first 3 runs (fast infrastructure check), then every 5 runs after that
  g) On kernel timeout: re-paste this cell next session. It will pull the
     latest results, skip completed runs, and continue.

If the first push fails, the script aborts loudly so you know within ~10 min
that something needs fixing (rather than discovering it after 12 hours).
================================================================================
"""

import os
import sys
import subprocess
import shutil
import stat
import time

# ---- CONFIG (edit only if your repo path differs) ---------------------------
REPO_USER = "ShussainML"
REPO_NAME = "eahm-fl-revision-experiments"
REPO_BRANCH = "main"
GIT_USER_NAME = "kaggle-runner"
GIT_USER_EMAIL = "kaggle-runner@users.noreply.github.com"
SECRET_NAME = "GITHUB_PAT"

WORK_DIR = "/kaggle/working"
REPO_DIR = f"{WORK_DIR}/{REPO_NAME}"
RESULTS_LOCAL_DIR = f"{WORK_DIR}/phase_a"     # where phase_a writes
RESULTS_REPO_DIR = f"{REPO_DIR}/results/phase_a"
DATA_ROOT = f"{WORK_DIR}/data"

# ---- 1. Load PAT from Kaggle secrets ---------------------------------------
def load_pat():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(SECRET_NAME)
    except Exception as e:
        print(f"!! Could not load secret '{SECRET_NAME}': {e}")
        print("   In Kaggle: Add-ons -> Secrets -> add GITHUB_PAT and attach to notebook")
        return None

PAT = load_pat()
if not PAT:
    raise RuntimeError(f"Secret {SECRET_NAME} not available; cannot proceed")
print(f"PAT loaded (len={len(PAT)}, prefix={PAT[:4]}...)")

# ---- 2. Clone or pull the repo using PAT in URL ----------------------------
def sh(cmd, check=True, mask=None):
    """Run a shell command; optionally mask a substring (PAT) in the output."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = p.stdout
    err = p.stderr
    if mask:
        out = out.replace(mask, "***")
        err = err.replace(mask, "***")
    if out: print(out, end="")
    if p.returncode != 0:
        print(f"  [rc={p.returncode}] {err[:400]}")
        if check:
            raise RuntimeError(f"command failed: {cmd[:80]}")
    return p.returncode

repo_url = f"https://{PAT}@github.com/{REPO_USER}/{REPO_NAME}.git"
safe_url = f"https://***@github.com/{REPO_USER}/{REPO_NAME}.git"

if os.path.isdir(REPO_DIR) and os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print(f"Repo exists; pulling latest from {safe_url}")
    sh(f"cd {REPO_DIR} && git fetch origin", check=False, mask=PAT)
    sh(f"cd {REPO_DIR} && git reset --hard origin/{REPO_BRANCH}", check=False, mask=PAT)
    sh(f"cd {REPO_DIR} && git pull --rebase --autostash origin {REPO_BRANCH}",
       check=False, mask=PAT)
else:
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    print(f"Cloning {safe_url}")
    sh(f"git clone {repo_url} {REPO_DIR}", check=True, mask=PAT)

# ---- 3. Configure git identity ---------------------------------------------
sh(f'cd {REPO_DIR} && git config user.name "{GIT_USER_NAME}"', check=False)
sh(f'cd {REPO_DIR} && git config user.email "{GIT_USER_EMAIL}"', check=False)

# ---- 4. Set up askpass so `git push` works without re-embedding PAT in URL --
# This avoids the PAT showing up in command history or git logs.
ASKPASS_PATH = f"{WORK_DIR}/.git_askpass.sh"
with open(ASKPASS_PATH, "w") as f:
    f.write(f"#!/bin/sh\necho '{PAT}'\n")
os.chmod(ASKPASS_PATH, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
os.environ["GIT_ASKPASS"] = ASKPASS_PATH
os.environ["GIT_TERMINAL_PROMPT"] = "0"

# Set the remote URL to a clean form (no embedded PAT); askpass supplies it
sh(f"cd {REPO_DIR} && git remote set-url origin "
   f"https://github.com/{REPO_USER}/{REPO_NAME}.git",
   check=False)

# Add .gitignore entries to keep .pt and cache files out of pushes
gitignore_path = os.path.join(REPO_DIR, ".gitignore")
existing = ""
if os.path.exists(gitignore_path):
    with open(gitignore_path) as f:
        existing = f.read()
needed = ["*.pt", "__pycache__/", ".ipynb_checkpoints/", "results/phase_a/ckpts/"]
to_add = [n for n in needed if n not in existing]
if to_add:
    with open(gitignore_path, "a") as f:
        f.write("\n" + "\n".join(to_add) + "\n")
    sh(f"cd {REPO_DIR} && git add .gitignore", check=False)

# ---- 5. Import the experiment module from the repo --------------------------
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))

# Make sure imports are fresh in case repo content changed across sessions
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("eahm_revision") or mod_name == "auto_push":
        del sys.modules[mod_name]

from auto_push import AutoPusher
from eahm_revision import phase_a

# ---- 6. Set up AutoPusher and run -------------------------------------------
os.makedirs(RESULTS_LOCAL_DIR, exist_ok=True)
os.makedirs(RESULTS_REPO_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

pusher = AutoPusher(
    repo_dir=REPO_DIR,
    results_local_dir=RESULTS_LOCAL_DIR,
    results_repo_dir=RESULTS_REPO_DIR,
    phase_name="phase_a",
    push_every=5,            # steady state: push every 5 completed runs
    fast_push_first_n=3,     # first 3 runs: push after every single one
                             # (validates push infrastructure quickly)
)

print("\n" + "="*72)
print("STARTING PHASE A")
print("="*72)
print(f"Repo:         {REPO_USER}/{REPO_NAME} (branch={REPO_BRANCH})")
print(f"Repo dir:     {REPO_DIR}")
print(f"Results dir:  {RESULTS_LOCAL_DIR} -> {RESULTS_REPO_DIR}")
print(f"Push policy:  every run for first 3, then every 5 runs")
print("="*72 + "\n")

t_start = time.time()
try:
    phase_a.main(
        results_dir=RESULTS_LOCAL_DIR,
        auto_pusher=pusher,
        data_root=DATA_ROOT,
        abort_if_first_push_fails=True,
    )
except KeyboardInterrupt:
    print("\n!! Interrupted by user; doing final push")
    pusher.final_push()
except Exception as e:
    print(f"\n!! Fatal error: {e}")
    import traceback; traceback.print_exc()
    print("\nAttempting final push to preserve whatever completed...")
    pusher.final_push()
    raise
finally:
    elapsed = time.time() - t_start
    print(f"\nSession elapsed: {elapsed/60:.1f} min")
    print(f"Pusher summary:  {pusher.summary()}")
    print(f"\nView results: https://github.com/{REPO_USER}/{REPO_NAME}/tree/{REPO_BRANCH}/results/phase_a")
