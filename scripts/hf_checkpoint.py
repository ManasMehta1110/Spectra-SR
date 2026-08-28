"""Checkpoint backup/restore via HF Hub -- wired up in Phase 0 per the plan (Section 2), before
the first real Colab Pro training run, rather than rediscovered under pressure after a session
disconnect (the failure mode hit on the GroundingBench project).

Usage:
    export SPECTRA_SR_HF_REPO=<your-username>/spectra-sr-checkpoints   # set before first use
    export HF_TOKEN=hf_...                                            # write-scoped token
    python scripts/hf_checkpoint.py check                     # verify auth + repo, upload nothing
    python scripts/hf_checkpoint.py push <local_checkpoint_dir>
    python scripts/hf_checkpoint.py pull <local_checkpoint_dir>

`check` exists because the only thing worse than having no backup is *believing* you have one.
Run it before starting a long job, not after it dies.
"""
import os
import sys

from huggingface_hub import HfApi, snapshot_download


def _repo_id() -> str:
    repo_id = os.environ.get("SPECTRA_SR_HF_REPO")
    if not repo_id:
        raise RuntimeError(
            "Set SPECTRA_SR_HF_REPO to a HF model repo (e.g. `your-username/spectra-sr-checkpoints`) "
            "before pushing/pulling checkpoints."
        )
    return repo_id


def check() -> bool:
    """Verify the token authenticates, the repo exists (creating it if not), and that a write
    actually succeeds -- by round-tripping a tiny sentinel file. Read-scoped tokens authenticate
    fine and only fail at the first real upload, which on Colab means discovering the problem
    ten hours in; this surfaces it in about two seconds."""
    repo_id = _repo_id()
    api = HfApi()
    who = api.whoami()
    print(f"authenticated as: {who.get('name', '?')}")
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    print(f"repo ok (private): {repo_id}")
    api.upload_file(path_or_fileobj=b"ok", path_in_repo=".backup_check",
                    repo_id=repo_id, repo_type="model")
    print("write test passed -- backups will work")
    return True


def push(local_dir: str) -> None:
    repo_id = _repo_id()
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    print(f"Pushed {local_dir} -> {repo_id}")


def push_files(paths, subdir: str = "") -> None:
    """Upload specific files rather than a whole directory. Used by the training loop's periodic
    backup: a run directory accumulates one checkpoint per epoch, and re-uploading all of them
    every epoch would waste most of the transfer on files that have not changed."""
    repo_id = _repo_id()
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    for path in paths:
        if not os.path.exists(path):
            continue
        dest = os.path.join(subdir, os.path.basename(path)).replace(os.sep, "/")
        api.upload_file(path_or_fileobj=path, path_in_repo=dest,
                        repo_id=repo_id, repo_type="model")


def backup_checkpoint(paths, subdir: str = "") -> bool:
    """Best-effort backup for use *inside* a training loop. Never raises.

    A failed upload -- expired token, dropped network, HF outage -- must not take down a
    multi-hour training run, because the run itself is the expensive thing and the backup is
    only insurance. Returns True on success so the caller can log the distinction; the failure
    is logged loudly rather than silently swallowed, since a backup that quietly stopped working
    is indistinguishable from one that never ran.
    """
    try:
        push_files(paths, subdir=subdir)
        return True
    except Exception as exc:  # noqa: BLE001 -- deliberate: see docstring
        print(f"[hf_checkpoint] WARNING: backup failed ({type(exc).__name__}: {exc}). "
              f"Training continues; this checkpoint exists only locally.")
        return False


def pull(local_dir: str) -> None:
    repo_id = _repo_id()
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
    print(f"Pulled {repo_id} -> {local_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        check()
        sys.exit(0)
    if len(sys.argv) != 3 or sys.argv[1] not in ("push", "pull"):
        print(__doc__)
        sys.exit(1)
    (push if sys.argv[1] == "push" else pull)(sys.argv[2])
