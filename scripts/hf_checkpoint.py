"""Checkpoint backup/restore via HF Hub -- wired up in Phase 0 per the plan (Section 2), before
the first real Colab Pro training run, rather than rediscovered under pressure after a session
disconnect (the failure mode hit on the GroundingBench project).

Usage:
    export SPECTRA_SR_HF_REPO=<your-username>/spectra-sr-checkpoints   # set before first use
    python scripts/hf_checkpoint.py push <local_checkpoint_dir>
    python scripts/hf_checkpoint.py pull <local_checkpoint_dir>
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


def push(local_dir: str) -> None:
    repo_id = _repo_id()
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    print(f"Pushed {local_dir} -> {repo_id}")


def pull(local_dir: str) -> None:
    repo_id = _repo_id()
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
    print(f"Pulled {repo_id} -> {local_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("push", "pull"):
        print(__doc__)
        sys.exit(1)
    (push if sys.argv[1] == "push" else pull)(sys.argv[2])
