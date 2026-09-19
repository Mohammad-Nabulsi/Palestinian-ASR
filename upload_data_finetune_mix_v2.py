from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi

REPO_ID = "mohammmeed/data_finetune_mix_v2"
REPO_TYPE = "dataset"

ROOT = Path("data_finetune_mix_v2")

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True)

files = sorted(
    [p for p in ROOT.rglob("*") if p.is_file()]
)

print(f"Found {len(files)} files to upload")

for p in tqdm(files, desc="Uploading files", unit="file"):
    rel = p.relative_to(ROOT)
    path_in_repo = rel.as_posix()

    tqdm.write(f"Uploading: {rel}")
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

print("Done.")
