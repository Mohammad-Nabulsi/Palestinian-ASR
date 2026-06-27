from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree


WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


RAW_LINKS = {
    "processed_qasr_segments": Path("processed_qasr_segments"),
    "omnilingual_apc": Path("omnilingual_selected") / "apc_north_levantine_all_splits",
    "casablanca_palestinian": Path("casablanca") / "levant" / "Palestine",
    "casablanca_jordanian": Path("casablanca") / "levant" / "Jordan",
}


LAYLA_SOURCE = Path("Layla") / "Layla Witheeb Jordanian Arabic Acoustic Dataset"
LAYLA_ALLOWED_SUFFIXES = {".wav", ".WAV", ".txt"}


def ensure_clean_symlink(link_path: Path, target_path: Path) -> None:
    relative_target = os.path.relpath(target_path, start=link_path.parent)
    if link_path.is_symlink():
        current_target = os.readlink(link_path)
        if current_target == relative_target:
            return
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(f"Refusing to replace existing non-symlink path: {link_path}")
    link_path.symlink_to(relative_target)


def extract_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as docx_zip:
        xml_bytes = docx_zip.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    paragraphs = []
    for para in root.iter(f"{WORD_NAMESPACE}p"):
        parts = []
        for node in para.iter(f"{WORD_NAMESPACE}t"):
            parts.append(node.text or "")
        if parts:
            paragraphs.append("".join(parts))
    return "\n".join(paragraphs).strip() + "\n"


def stage_layla(output_root: Path) -> None:
    layla_output = output_root / "layla"
    layla_output.mkdir(parents=True, exist_ok=True)

    copied = 0
    converted = 0
    for source_path in sorted(LAYLA_SOURCE.rglob("*")):
        if not source_path.is_file():
            continue

        relative_path = source_path.relative_to(LAYLA_SOURCE)
        if source_path.suffix in LAYLA_ALLOWED_SUFFIXES:
            destination = layla_output / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            copied += 1
        elif source_path.suffix == ".docx":
            destination = layla_output / relative_path.with_suffix(".txt")
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(extract_docx_text(source_path), encoding="utf-8")
            converted += 1

    print(f"Layla staged to {layla_output}")
    print(f"Copied {copied} audio/text files")
    print(f"Converted {converted} Word files to txt")



def main() -> None:
    parser = argparse.ArgumentParser(description="Stage raw datasets into a single root data/ directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Root staging directory.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    masc_filtered = output_root / "masc_c_only"
    if not masc_filtered.exists():
        raise SystemExit(
            f"Expected filtered MASC dataset at {masc_filtered}. Run scripts/filter_masc_c_only.py first."
        )

    for name, rel_target in RAW_LINKS.items():
        target_path = (root / rel_target).resolve()
        if not target_path.exists():
            raise SystemExit(f"Missing source dataset: {target_path}")
        ensure_clean_symlink(output_root / name, target_path)
        print(f"Linked {name} -> {target_path}")

    stage_layla(output_root)


if __name__ == "__main__":
    main()
