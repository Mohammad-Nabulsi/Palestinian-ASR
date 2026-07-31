#!/usr/bin/env python3
"""Replace every symlink under data/ with the real file it points at, by rename.

Both trees live on the same MooseFS device, so os.rename is a metadata operation:
no bytes are copied and the quota never sees a transient doubling. After this runs,
data/ owns the shards outright and data_cleaned_text_merged_v1/ is left holding only
the shards data/ deliberately does not reference (the superseded Omnilingual pass).

Crash safety comes from the manifest: it is written and fsynced before any rename,
and it records (link, target) for every move. A rerun replays it, skipping entries
that already landed. Do not regenerate the manifest after a partial run -- the
symlinks it was derived from will be gone.
"""

import argparse
import json
import os
import sys

DATA_ROOT = "/workspace/asr/Palestinian-ASR/data"
MERGED_ROOT = "/workspace/asr/Palestinian-ASR/data_cleaned_text_merged_v1"


def build_manifest(data_root, merged_root):
    entries = []
    for dirpath, _, filenames in os.walk(data_root):
        for name in filenames:
            link = os.path.join(dirpath, name)
            if not os.path.islink(link):
                continue
            target = os.readlink(link)
            if not target.startswith(merged_root + os.sep):
                raise SystemExit(f"unexpected symlink target outside merged root: {link} -> {target}")
            entries.append({"link": link, "target": target})
    entries.sort(key=lambda e: e["link"])
    return entries


def verify_same_device(data_root, merged_root):
    a = os.stat(data_root).st_dev
    b = os.stat(merged_root).st_dev
    if a != b:
        raise SystemExit(
            f"refusing to run: {data_root} (dev {a}) and {merged_root} (dev {b}) are on "
            "different filesystems, so rename would fall back to a full copy"
        )
    return a


def move_one(link, target):
    """Return 'moved', 'already', or raise. Idempotent per entry."""
    if os.path.islink(link):
        if not os.path.isfile(target):
            raise RuntimeError(f"symlink present but target missing: {link} -> {target}")
        os.unlink(link)
        os.rename(target, link)
        return "moved"

    if os.path.isfile(link):
        # Already materialized on a previous pass.
        return "already"

    # Interrupted between unlink and rename: the mapping is identity, so the
    # target still tells us where the file is.
    if os.path.isfile(target):
        os.rename(target, link)
        return "moved"

    raise RuntimeError(f"neither link nor target exists: {link} <- {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--merged-root", default=MERGED_ROOT)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--execute", action="store_true", help="without this, only plan and write the manifest")
    args = ap.parse_args()

    dev = verify_same_device(args.data_root, args.merged_root)

    if os.path.exists(args.manifest):
        with open(args.manifest) as fh:
            entries = json.load(fh)
        print(f"resuming from existing manifest: {len(entries)} entries", flush=True)
    else:
        entries = build_manifest(args.data_root, args.merged_root)
        with open(args.manifest, "w") as fh:
            json.dump(entries, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"wrote manifest: {len(entries)} entries", flush=True)

    print(f"device {dev} shared by both roots -> renames are metadata-only", flush=True)

    by_dir = {}
    for e in entries:
        by_dir[os.path.dirname(e["link"])] = by_dir.get(os.path.dirname(e["link"]), 0) + 1
    for d in sorted(by_dir):
        print(f"  {by_dir[d]:>5}  {os.path.relpath(d, args.data_root)}", flush=True)

    if not args.execute:
        print("\ndry run -- pass --execute to perform the renames", flush=True)
        return 0

    moved = already = 0
    for i, e in enumerate(entries, 1):
        result = move_one(e["link"], e["target"])
        if result == "moved":
            moved += 1
        else:
            already += 1
        if i % 250 == 0 or i == len(entries):
            print(f"  {i}/{len(entries)}  moved={moved} already={already}", flush=True)

    print(f"\ndone: {moved} renamed, {already} already in place", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
