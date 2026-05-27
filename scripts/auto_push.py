"""
AutoPusher — incremental commit & push helper for long-running Kaggle experiments.

Designed for the EAHM-FL revision experiments where a Kaggle session may die
after ~12 hours. AutoPusher pushes intermediate results to GitHub every N
completed runs so that the next session can resume from the latest push.

CONTRACT:
  - Caller copies new CSV/JSON/PNG files into self.results_local_dir as runs complete.
  - Caller calls self.run_completed() once per completed (algo, alpha, seed) run.
  - AutoPusher copies results into the repo working tree and pushes when
    self.completed_since_last_push >= push_every.
  - On push failure: logs error, returns False, does NOT raise. Training continues.

ASSUMPTIONS:
  - Caller has configured git in self.repo_dir with a credential helper
    (e.g. GIT_ASKPASS pointing at a script that emits the PAT). Bootstrap
    cell handles this.
  - Caller has set git user.email and user.name in self.repo_dir.
"""

import os
import shutil
import subprocess
import time


class AutoPusher:
    def __init__(self, repo_dir, results_local_dir, results_repo_dir,
                 phase_name, push_every=5, fast_push_first_n=3):
        """
        repo_dir            : git working tree, e.g. /kaggle/working/eahm-fl-revision-experiments
        results_local_dir   : where the experiment writes new CSVs, e.g. /kaggle/working/phase_a
        results_repo_dir    : mirror inside the repo, e.g. {repo_dir}/results/phase_a
        phase_name          : 'phase_a' (used in commit messages)
        push_every          : push after this many completed runs (steady state)
        fast_push_first_n   : for the first N runs, push after every single run
                              (used to validate the push infrastructure quickly)
        """
        self.repo_dir = repo_dir
        self.results_local_dir = results_local_dir
        self.results_repo_dir = results_repo_dir
        self.phase_name = phase_name
        self.push_every = push_every
        self.fast_push_first_n = fast_push_first_n

        self.completed_total = 0
        self.completed_since_last_push = 0
        self.total_pushes = 0
        self.successful_pushes = 0
        self.last_push_status = "not_yet"
        self.last_push_error = None

        os.makedirs(self.results_local_dir, exist_ok=True)
        os.makedirs(self.results_repo_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Restore (call once at session start)
    # -------------------------------------------------------------------

    def restore_from_repo(self):
        """Copy any existing results from repo into the local working dir.
        Returns the number of files restored.
        """
        if not os.path.isdir(self.results_repo_dir):
            return 0
        count = 0
        for root, dirs, files in os.walk(self.results_repo_dir):
            rel = os.path.relpath(root, self.results_repo_dir)
            local_root = (self.results_local_dir if rel == "."
                          else os.path.join(self.results_local_dir, rel))
            os.makedirs(local_root, exist_ok=True)
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(local_root, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    count += 1
        return count

    # -------------------------------------------------------------------
    # Per-run hooks
    # -------------------------------------------------------------------

    def run_completed(self):
        """Caller signals one (algo, alpha, seed) run has completed."""
        self.completed_total += 1
        self.completed_since_last_push += 1

    def maybe_push(self):
        """Push if we've accumulated enough completed runs. Returns True if a
        push was attempted (regardless of outcome)."""
        threshold = (1 if self.completed_total <= self.fast_push_first_n
                     else self.push_every)
        if self.completed_since_last_push >= threshold:
            ok = self._do_push(label=f"auto-{self.completed_total}")
            self.completed_since_last_push = 0
            return True, ok
        return False, None

    def final_push(self):
        """Always-runs push at end of session (or end of experiment)."""
        return self._do_push(label="final")

    # -------------------------------------------------------------------
    # Internal: copy & push
    # -------------------------------------------------------------------

    def _copy_local_to_repo(self):
        """Mirror local results dir into the repo working tree."""
        if not os.path.isdir(self.results_local_dir):
            return
        for root, dirs, files in os.walk(self.results_local_dir):
            rel = os.path.relpath(root, self.results_local_dir)
            repo_root = (self.results_repo_dir if rel == "."
                         else os.path.join(self.results_repo_dir, rel))
            os.makedirs(repo_root, exist_ok=True)
            for f in files:
                # Skip model checkpoints — too large for GitHub
                if f.endswith(".pt"):
                    continue
                src = os.path.join(root, f)
                dst = os.path.join(repo_root, f)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[auto_push] copy failed for {src}: {e}")

    def _do_push(self, label="auto"):
        """Stage results, commit, push. Returns True on success."""
        self.total_pushes += 1
        t0 = time.time()
        try:
            self._copy_local_to_repo()

            # git add
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "add", f"results/{self.phase_name}/"],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                self.last_push_status = "add_failed"
                self.last_push_error = r.stderr[:300]
                print(f"[auto_push] git add failed: {r.stderr[:200]}")
                return False

            # check if anything is staged
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "diff", "--cached", "--name-only"],
                capture_output=True, text=True
            )
            if not r.stdout.strip():
                self.last_push_status = "nothing_to_commit"
                return True  # not a failure

            # commit
            msg = f"phase_a [{label}] completed={self.completed_total}"
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "commit", "-m", msg],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                self.last_push_status = "commit_failed"
                self.last_push_error = r.stderr[:300]
                print(f"[auto_push] git commit failed: {r.stderr[:200]}")
                return False

            # pull --rebase to avoid trivial conflicts with other sessions
            subprocess.run(
                ["git", "-C", self.repo_dir, "pull", "--rebase", "--autostash",
                 "origin", "main"],
                capture_output=True, text=True
            )

            # push
            r = subprocess.run(
                ["git", "-C", self.repo_dir, "push", "origin", "main"],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                self.last_push_status = "push_failed"
                self.last_push_error = r.stderr[:500]
                print(f"[auto_push] git push failed: {r.stderr[:300]}")
                return False

            self.successful_pushes += 1
            elapsed = time.time() - t0
            self.last_push_status = f"ok ({elapsed:.1f}s)"
            print(f"[auto_push] push #{self.total_pushes} OK "
                  f"after {self.completed_total} runs ({elapsed:.1f}s)")
            return True

        except Exception as e:
            self.last_push_status = "exception"
            self.last_push_error = str(e)[:300]
            print(f"[auto_push] exception: {e}")
            return False

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def summary(self):
        return (f"completed_runs={self.completed_total} "
                f"push_attempts={self.total_pushes} "
                f"successful={self.successful_pushes} "
                f"last_status={self.last_push_status}")
