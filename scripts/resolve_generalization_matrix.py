#!/usr/bin/env python
"""Resolve the 37-checkpoint x domain-list generalization-eval matrix into concrete adapter
paths, by reading each run's own SUMMARY.json (never hand-typing checkpoint paths). Validates
every referenced directory actually exists before writing the matrix out, so a bad path fails
loudly here instead of silently mid-sweep.

Output: /root/Palestinian-ASR/generalization_matrix.json -- a list of
    {"name": ..., "merge_adapters": [...], "final_adapter": str|None, "domains": [...]}
consumed by run_generalization_eval.py.
"""
import json
from pathlib import Path

RUNS_DIR = Path("/workspace/asr/Palestinian-ASR/Runs/whisper_medium_pal")
CKPT_ROOT = Path("/workspace/asr_env/checkpoints")

ALL_DOMAINS = ["omni_test", "layla_test", "casa_jor_test", "qasr_lev_2h", "masc_lev_2h",
               "qasr_non_lev_2h", "masc_non_lev_2h"]


def domains_minus(*seen):
    return [d for d in ALL_DOMAINS if d not in seen]


def summary(run_tag):
    p = RUNS_DIR / run_tag / "SUMMARY.json"
    return json.loads(p.read_text())


def final_dir(d, tag, earlystop=True):
    """Mirrors _final_adapter_dir from the training script: early-stop runs use best_dir;
    fixed multi-epoch runs (named *_s2epN or *_s1ep2_s2ep2) use last_epoch_dir."""
    t = d["train"][tag]
    return t["best_dir"] if earlystop else (t.get("last_epoch_dir") or t["best_dir"])


def ckpt_dir(pal_run):
    return CKPT_ROOT / f"whisper_medium_pal__{pal_run}"


def explicit_epoch(pal_run, epoch_n):
    """The EXPLICIT, immutable per-epoch checkpoint (stage1/epochNNN), never the mutable
    "best" pointer. Needed specifically for run2/run3/run4's original 1-epoch-only runs:
    their stage-1 checkpoint namespace was later REUSED (same PAL_RUN, same CKPT_DIR) by the
    _stage1ep2 continuation, which overwrote "best" whenever epoch 2 beat epoch 1 -- so
    reading "best_dir" today can silently return epoch 2's weights for what should be the
    epoch-1-only checkpoint. Verified via md5: run2_jor's & run4_omni_jor's "best" now equals
    their epoch002 (contaminated); run3_omni's happens to still equal epoch001 (lucky, not
    reliable). epochNNN dirs are written once and never touched again, so they're safe."""
    return str(ckpt_dir(pal_run) / "openai__whisper-medium" / "stage1" / f"epoch{epoch_n:03d}")


def stage1_ckpt(run_tag, tag="stage1"):
    """The actual stage-1 (or stage2_pretrain, for run11) adapter path, read directly from
    that run's own SUMMARY.json train.<tag>.best_dir -- NOT a guessed "stage1_merged_adapter"
    directory-naming convention. Several pre-existing runs (run2/3/4's _stage1ep2 variants)
    resumed their 1-epoch sibling's SHARED checkpoint namespace to reach epoch 2 rather than
    using their own, so a naming-convention guess silently resolves to the WRONG (1-epoch)
    checkpoint for those -- verified via matching md5sums between run2_jor's and the guessed
    run2_jor_stage1ep2's "stage1_merged_adapter" files. Reading best_dir straight from the
    SUMMARY.json is what that run's own "stage1_merged" eval actually scored, so it's
    guaranteed consistent with the WER numbers already reported for it."""
    return summary(run_tag)["train"][tag]["best_dir"]


checkpoints = []

# ---- run2 (Jordanian) -- trained: casa_jor ----
d1, d1e = domains_minus("casa_jor_test"), domains_minus("casa_jor_test")
checkpoints += [
    {"name": "run2_1ep_alone", "merge_adapters": [explicit_epoch("run2_jor", 1)],
     "final_adapter": None, "domains": d1},
    {"name": "run2_2ep_alone", "merge_adapters": [stage1_ckpt("run2_jor_stage1ep2_earlystop")],
     "final_adapter": None, "domains": d1},
    {"name": "run2_1ep_stage2final",
     "merge_adapters": [explicit_epoch("run2_jor", 1)],
     "final_adapter": final_dir(summary("run2_jor_earlystop"), "stage2", True), "domains": d1},
    {"name": "run2_2ep_stage2final",
     "merge_adapters": [stage1_ckpt("run2_jor_stage1ep2_earlystop")],
     "final_adapter": final_dir(summary("run2_jor_stage1ep2_earlystop"), "stage2", True), "domains": d1},
]

# ---- run3 (Omni) -- trained: omni ----
d3 = domains_minus("omni_test")
checkpoints += [
    {"name": "run3_1ep_alone", "merge_adapters": [explicit_epoch("run3_omni", 1)],
     "final_adapter": None, "domains": d3},
    {"name": "run3_2ep_alone", "merge_adapters": [stage1_ckpt("run3_omni_stage1ep2_earlystop")],
     "final_adapter": None, "domains": d3},
    {"name": "run3_1ep_stage2final",
     "merge_adapters": [explicit_epoch("run3_omni", 1)],
     "final_adapter": final_dir(summary("run3_omni_earlystop"), "stage2", True), "domains": d3},
    {"name": "run3_2ep_stage2final",
     "merge_adapters": [stage1_ckpt("run3_omni_stage1ep2_earlystop")],
     "final_adapter": final_dir(summary("run3_omni_stage1ep2_earlystop"), "stage2", True), "domains": d3},
]

# ---- run4 (Omni+Jordanian) -- trained: omni, casa_jor ----
d4 = domains_minus("omni_test", "casa_jor_test")
checkpoints += [
    {"name": "run4_1ep_alone", "merge_adapters": [explicit_epoch("run4_omni_jor", 1)],
     "final_adapter": None, "domains": d4},
    {"name": "run4_2ep_alone", "merge_adapters": [stage1_ckpt("run4_omni_jor_stage1ep2_earlystop")],
     "final_adapter": None, "domains": d4},
    {"name": "run4_1ep_stage2final",
     "merge_adapters": [explicit_epoch("run4_omni_jor", 1)],
     "final_adapter": final_dir(summary("run4_omni_jor_earlystop"), "stage2", True), "domains": d4},
    {"name": "run4_2ep_stage2final",
     "merge_adapters": [stage1_ckpt("run4_omni_jor_stage1ep2_earlystop")],
     "final_adapter": final_dir(summary("run4_omni_jor_stage1ep2_earlystop"), "stage2", True), "domains": d4},
]

# ---- run5 (layla+jor, NOT built by me) -- trained: layla, casa_jor ----
d5 = domains_minus("layla_test", "casa_jor_test")
checkpoints += [
    {"name": "run5_1ep_alone", "merge_adapters": [stage1_ckpt("run5_layla_jor_earlystop")],
     "final_adapter": None, "domains": d5},
    {"name": "run5_2ep_alone", "merge_adapters": [stage1_ckpt("run5_layla_jor_s1ep2")],
     "final_adapter": None, "domains": d5},
    {"name": "run5_1ep_stage2final",
     "merge_adapters": [stage1_ckpt("run5_layla_jor_earlystop")],
     "final_adapter": final_dir(summary("run5_layla_jor_earlystop"), "stage2", True), "domains": d5},
    {"name": "run5_2ep_stage2final",
     "merge_adapters": [stage1_ckpt("run5_layla_jor_s1ep2")],
     "final_adapter": final_dir(summary("run5_layla_jor_s1ep2_s2ep2_s2ep2"), "stage2", False), "domains": d5},
]

# ---- run6 (qasr_lev 10h) -- trained: qasr_lev ----
d6 = domains_minus("qasr_lev_2h")
checkpoints += [
    {"name": "run6_1ep_alone", "merge_adapters": [stage1_ckpt("run6_qasr_lev_10h_earlystop")],
     "final_adapter": None, "domains": d6},
    {"name": "run6_2ep_alone", "merge_adapters": [stage1_ckpt("run6_qasr_lev_10h_ep2_s1ep2_earlystop")],
     "final_adapter": None, "domains": d6},
    {"name": "run6_1ep_stage2final",
     "merge_adapters": [stage1_ckpt("run6_qasr_lev_10h_earlystop")],
     "final_adapter": final_dir(summary("run6_qasr_lev_10h_earlystop"), "stage2", True), "domains": d6},
    {"name": "run6_2ep_stage2final",
     "merge_adapters": [stage1_ckpt("run6_qasr_lev_10h_ep2_s1ep2_earlystop")],
     "final_adapter": final_dir(summary("run6_qasr_lev_10h_ep2_s1ep2_earlystop"), "stage2", True), "domains": d6},
]

# ---- run7 (qasr_non_lev 50h) -- trained: qasr_non_lev ----
d7 = domains_minus("qasr_non_lev_2h")
checkpoints += [
    {"name": "run7_1ep_alone", "merge_adapters": [stage1_ckpt("run7_qasr_non_lev_50h_earlystop")],
     "final_adapter": None, "domains": d7},
    {"name": "run7_2ep_alone", "merge_adapters": [stage1_ckpt("run7_qasr_non_lev_50h_ep2_s1ep2_earlystop")],
     "final_adapter": None, "domains": d7},
    {"name": "run7_1ep_stage2final",
     "merge_adapters": [stage1_ckpt("run7_qasr_non_lev_50h_earlystop")],
     "final_adapter": final_dir(summary("run7_qasr_non_lev_50h_earlystop"), "stage2", True), "domains": d7},
    {"name": "run7_2ep_stage2final",
     "merge_adapters": [stage1_ckpt("run7_qasr_non_lev_50h_ep2_s1ep2_earlystop")],
     "final_adapter": final_dir(summary("run7_qasr_non_lev_50h_ep2_s1ep2_earlystop"), "stage2", True), "domains": d7},
]

# ---- run8 (masc_lev+qasr_lev) -- trained: masc_lev, qasr_lev ----
d8 = domains_minus("masc_lev_2h", "qasr_lev_2h")
checkpoints += [
    {"name": "run8_1ep_alone", "merge_adapters": [stage1_ckpt("run8_qasr_lev_masc_lev_earlystop")],
     "final_adapter": None, "domains": d8},
    {"name": "run8_1ep_stage2final",
     "merge_adapters": [stage1_ckpt("run8_qasr_lev_masc_lev_earlystop")],
     "final_adapter": final_dir(summary("run8_qasr_lev_masc_lev_earlystop"), "stage2", True), "domains": d8},
]
# run8's ep2 checkpoints: only add if that run has finished (SUMMARY.json exists)
# NOTE: stage1_merged_adapter/best are legitimately EMPTY for the seeded ep2 continuations --
# the seed copy only carries over the resumable ckpt_step (for native resume), never the
# source run's own "best" files, and epoch 2 here didn't beat epoch 1's val WER, so nothing
# ever wrote to this run's own best_dir. The in-memory merge_and_unload() at end of training
# used the final (epoch 2) weights though, so stage1/epoch002 is the correct on-disk stand-in.
_run8ep2_ckpt = Path(explicit_epoch("run8_qasr_lev_masc_lev_ep2", 2))
if _run8ep2_ckpt.exists():
    checkpoints.append({"name": "run8_2ep_alone", "merge_adapters": [str(_run8ep2_ckpt)],
                        "final_adapter": None, "domains": d8})
_run8ep2_summary_path = RUNS_DIR / "run8_qasr_lev_masc_lev_ep2_s1ep2_earlystop" / "SUMMARY.json"
if _run8ep2_summary_path.exists():
    checkpoints.append({
        "name": "run8_2ep_stage2final", "merge_adapters": [str(_run8ep2_ckpt)],
        "final_adapter": final_dir(json.loads(_run8ep2_summary_path.read_text()), "stage2", True),
        "domains": d8})

# ---- run9 (masc_lev+qasr_lev+qasr_non_lev) -- trained: masc_lev, qasr_lev, qasr_non_lev ----
d9 = domains_minus("masc_lev_2h", "qasr_lev_2h", "qasr_non_lev_2h")
checkpoints += [
    {"name": "run9_1ep_alone",
     "merge_adapters": [stage1_ckpt("run9_qasr_lev_masc_lev_nonlev50h_earlystop")],
     "final_adapter": None, "domains": d9},
    {"name": "run9_1ep_stage2final",
     "merge_adapters": [stage1_ckpt("run9_qasr_lev_masc_lev_nonlev50h_earlystop")],
     "final_adapter": final_dir(summary("run9_qasr_lev_masc_lev_nonlev50h_earlystop"), "stage2", True),
     "domains": d9},
]
# Same on-disk gap as run8_ep2 above (seed copy never carries over "best", and epoch 2 here
# didn't beat epoch 1's val WER) -- use the immutable epoch002 checkpoint instead.
_run9ep2_ckpt = Path(explicit_epoch("run9_qasr_lev_masc_lev_nonlev50h_ep2", 2))
if _run9ep2_ckpt.exists():
    checkpoints.append({"name": "run9_2ep_alone", "merge_adapters": [str(_run9ep2_ckpt)],
                        "final_adapter": None, "domains": d9})
_run9ep2_summary_path = RUNS_DIR / "run9_qasr_lev_masc_lev_nonlev50h_ep2_s1ep2_earlystop" / "SUMMARY.json"
if _run9ep2_summary_path.exists():
    checkpoints.append({
        "name": "run9_2ep_stage2final", "merge_adapters": [str(_run9ep2_ckpt)],
        "final_adapter": final_dir(json.loads(_run9ep2_summary_path.read_text()), "stage2", True),
        "domains": d9})

# ---- run10 (all 6 combined) -- trained: everything except masc_non_lev ----
d10 = domains_minus("omni_test", "layla_test", "casa_jor_test", "qasr_lev_2h", "masc_lev_2h",
                     "qasr_non_lev_2h")
checkpoints += [
    {"name": "run10_alone", "merge_adapters": [stage1_ckpt("run10_all_combined_earlystop")],
     "final_adapter": None, "domains": d10},
    {"name": "run10_stage2final",
     "merge_adapters": [stage1_ckpt("run10_all_combined_earlystop")],
     "final_adapter": final_dir(summary("run10_all_combined_earlystop"), "stage2", True), "domains": d10},
]

# ---- run11 (staged: qasr_non_lev -> merge -> masc_lev+qasr_lev -> merge -> pal) ----
r11 = summary("run11_qasr_nonlev_then_masc_qasr_lev")
r11_ckpt = ckpt_dir("run11_qasr_nonlev_then_masc_qasr_lev")
d11_stage1 = domains_minus("qasr_non_lev_2h")  # only qasr_non_lev seen so far
d11_full = domains_minus("qasr_non_lev_2h", "masc_lev_2h", "qasr_lev_2h")  # all 3 pretrain corpora seen
checkpoints += [
    {"name": "run11_stage1_alone", "merge_adapters": [str(r11_ckpt / "stage1_merged_adapter")],
     "final_adapter": None, "domains": d11_stage1},
    {"name": "run11_stage2_alone",
     "merge_adapters": [str(r11_ckpt / "stage1_merged_adapter"), str(r11_ckpt / "stage2_merged_adapter")],
     "final_adapter": None, "domains": d11_full},
    {"name": "run11_stage3final",
     "merge_adapters": [str(r11_ckpt / "stage1_merged_adapter"), str(r11_ckpt / "stage2_merged_adapter")],
     "final_adapter": final_dir(r11, "stage3", True), "domains": d11_full},
]

# ---- sanity check: every 1ep/2ep pair within a family must resolve to DIFFERENT weights ----
# (this is exactly the class of bug caught above -- a shared/mutated checkpoint namespace
# silently making two supposedly-different checkpoints identical). Cheap enough to always run.
import hashlib as _hashlib
def _adapter_hash(dir_path):
    f = Path(dir_path) / "adapter_model.safetensors"
    return _hashlib.md5(f.read_bytes()).hexdigest() if f.exists() else None

by_prefix = {}
for c in checkpoints:
    if "_1ep_" in c["name"] or "_2ep_" in c["name"]:
        prefix, kind = c["name"].rsplit("_", 2)[0], c["name"].split("_")[1] + "ep"
        # e.g. "run2_1ep_alone" -> family="run2", suffix="alone"; "run2_2ep_alone" -> family="run2"
        parts = c["name"].split("_")
        family = parts[0]
        suffix = "_".join(parts[2:])  # "alone" or "stage2final"
        by_prefix.setdefault((family, suffix), {})[parts[1]] = c

collisions = []
for (family, suffix), variants in by_prefix.items():
    if "1ep" not in variants or "2ep" not in variants:
        continue
    h1 = _adapter_hash(variants["1ep"]["merge_adapters"][-1])
    h2 = _adapter_hash(variants["2ep"]["merge_adapters"][-1])
    if h1 is not None and h1 == h2:
        collisions.append((variants["1ep"]["name"], variants["2ep"]["name"]))
if collisions:
    print(f"COLLISION: {len(collisions)} 1ep/2ep pairs resolve to IDENTICAL adapter weights:")
    for a, b in collisions:
        print(f"  {a} == {b}")
    raise SystemExit(1)
else:
    print(f"Distinctness OK: all {len(by_prefix)} 1ep/2ep families resolve to different weights.")

# ---- validate every referenced path exists AND actually contains adapter weights ----
# (a bare Path.exists() check on the directory is not enough -- the seeded-ep2 bug below
# left "best"/"stage1_merged_adapter" dirs present-but-empty, which silently passed a
# directory-only check while every load attempt at eval time crashed on the missing file)
missing = []
for c in checkpoints:
    for p in c["merge_adapters"]:
        if not (Path(p) / "adapter_model.safetensors").exists():
            missing.append((c["name"], p))
    if c["final_adapter"] and not (Path(c["final_adapter"]) / "adapter_model.safetensors").exists():
        missing.append((c["name"], c["final_adapter"]))
if missing:
    print(f"MISSING {len(missing)} paths:")
    for name, p in missing:
        print(f"  {name}: {p}")
else:
    print(f"All paths OK for {len(checkpoints)} checkpoints.")

out = Path("/root/Palestinian-ASR/generalization_matrix.json")
out.write_text(json.dumps(checkpoints, indent=2))
total_evals = sum(len(c["domains"]) for c in checkpoints)
print(f"Wrote {len(checkpoints)} checkpoints, {total_evals} total domain-evals -> {out}")
