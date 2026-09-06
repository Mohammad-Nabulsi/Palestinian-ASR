from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "casablanca"

# Requested grouping:
# - far dialects: Algeria, Mauritania, Morocco, Yemen
# - levant: Palestine, Jordan
# - relevant arabic: everything else in the dataset root
GROUPS = {
    "far_dialects": ["Algeria", "Mauritania", "Morocco", "Yemen"],
    "levant": ["Palestine", "Jordan"],
}


def move_folder(src: Path, dst_root: Path) -> None:
    if not src.exists():
        print(f"skip missing: {src.name}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name

    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")

    print(f"move {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    shutil.move(str(src), str(dst))


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(ROOT)

    for group_name, folders in GROUPS.items():
        group_dir = ROOT / group_name
        for folder in folders:
            move_folder(ROOT / folder, group_dir)

    relevant_dir = ROOT / "relevant_arabic"
    relevant_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.name in GROUPS:
            continue
        if child.name == "relevant_arabic":
            continue
        if child.name == ".cache":
            continue

        print(f"move {child.relative_to(ROOT)} -> relevant_arabic/{child.name}")
        target = relevant_dir / child.name
        if target.exists():
            raise FileExistsError(f"destination already exists: {target}")
        shutil.move(str(child), str(target))

    print("done")


if __name__ == "__main__":
    main()
