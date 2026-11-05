"""Deploy the demo to the Hugging Face Space (run by the owner; creates a PUBLIC Space).

    uv run python -m plumescout.demo          # build app/assets first
    uv run python app/deploy_space.py

Uploads app.py, README.md, requirements.txt and assets/ only (no credentials, no raw data).
Uses HF_TOKEN from .env.
"""

from __future__ import annotations

import os

from huggingface_hub import HfApi

from plumescout.config import load_env

REPO = "Muhammadshahidusman/plumescout"

if __name__ == "__main__":
    load_env()
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.repo_exists(REPO, repo_type="space"):
        print(f"Space {REPO} exists, uploading files only")
    else:
        api.create_repo(REPO, repo_type="space", space_sdk="gradio", private=False)
    api.upload_folder(
        folder_path=os.path.dirname(os.path.abspath(__file__)),
        repo_id=REPO,
        repo_type="space",
        allow_patterns=["app.py", "README.md", "requirements.txt", "assets/**"],
        ignore_patterns=["**/__pycache__/**"],
        commit_message="PlumeScout demo: precomputed Gulf panels (non-commercial)",
    )
    print(f"Deployed: https://huggingface.co/spaces/{REPO}")
