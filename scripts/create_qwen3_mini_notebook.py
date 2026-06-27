import copy
import json
from pathlib import Path


SRC = Path("Runs/qwen3_asr_0_6b/qwen3_asr_0_6b_lora_run.ipynb")
DST_DIR = Path("Runs/qwen3_asr_0_6b_mini_50h_100eval")
DST = DST_DIR / "qwen3_asr_0_6b_lora_mini_50h_100eval_run.ipynb"
RUN_DIR = Path("/home/MohammadNabulsi/whisper/Runs/qwen3_asr_0_6b_mini_50h_100eval")
CACHE_DIR = Path("/home/MohammadNabulsi/whisper/cache/qwen3_asr_0_6b_mini_50h_100eval")


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    nb = copy.deepcopy(nb)

    set_source(
        nb["cells"][0],
        """# Qwen3-ASR 0.6B LoRA Mini Run: 50h Train / 100-Sample Evals

This notebook is a deliberately smaller, separate run derived from `qwen3_asr_0_6b_lora_run.ipynb`.

Differences from the current/full run:
- unique run directory: `/home/MohammadNabulsi/whisper/Runs/qwen3_asr_0_6b_mini_50h_100eval`
- unique cache directory: `/home/MohammadNabulsi/whisper/cache/qwen3_asr_0_6b_mini_50h_100eval`
- fresh manifest rebuild
- deterministic training subset capped at 50 total audio hours
- validation/test/baseline generation capped at 100 samples
- final integrity report rejects empty prediction files
""",
    )

    config = """from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
MINI_RUN_ID = "qwen3_asr_0_6b_mini_50h_100eval"

RUN_DIR = Path("/home/MohammadNabulsi/whisper/Runs/qwen3_asr_0_6b_mini_50h_100eval")
NOTEBOOK_PATH = RUN_DIR / "qwen3_asr_0_6b_lora_mini_50h_100eval_run.ipynb"
LOG_DIR = RUN_DIR / "logs"
OUTPUT_DIR = RUN_DIR / "checkpoints"
BEST_MODEL_DIR = RUN_DIR / "best"
MANIFEST_DIR = RUN_DIR / "manifests"
CACHE_DIR = Path("/home/MohammadNabulsi/whisper/cache/qwen3_asr_0_6b_mini_50h_100eval")

TRAIN_SOURCE_DIRS = [
    Path("/home/MohammadNabulsi/whisper/casablanca/relevant_arabic"),
    Path("/home/MohammadNabulsi/whisper/omnilingual_selected/other_arabic_dialects"),
]

HELDOUT_VAL_TEST_SOURCE_DIRS = [
    Path("/home/MohammadNabulsi/whisper/omnilingual_selected/apc_north_levantine_all_splits"),
    Path("/home/MohammadNabulsi/whisper/casablanca/levant"),
]

# Optional examples for small tests only, not the full source list.
OMNILINGUAL_ARROW_EXAMPLE = Path("/home/MohammadNabulsi/whisper/omnilingual_selected/apc_north_levantine_all_splits/data-00000-of-00003.arrow")
CASABLANCA_PARQUET_EXAMPLE = Path("/home/MohammadNabulsi/whisper/casablanca/levant/Palestine/test-00001-of-00002.parquet")
QASR_WAV_EXAMPLE = Path("/home/MohammadNabulsi/whisper/QASR/alt/alt/arabic-speech-web/mgb2.1/wav/0A4D5AA5-CA9E-4EB5-99D8-BD6D6DA2C58C.wav")
QASR_XML_EXAMPLE = Path("/home/MohammadNabulsi/whisper/QASR/mgb2.1/release/train_20210109/xml/0A4D5AA5-CA9E-4EB5-99D8-BD6D6DA2C58C.xml")
QASR_ROOT = Path("/home/MohammadNabulsi/whisper/QASR")
INCLUDE_QASR_IN_FULL_MANIFEST = False

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 120.0
MIN_AUDIO_SECONDS = 0.3

SPLIT_SEED = 42
FORCE_RESPLIT = True
VAL_FRACTION_OF_UNSPLIT_HELDOUT = 0.50

MINI_MAX_TRAIN_HOURS = 50.0
MINI_MAX_EVAL_SAMPLES = 100
MINI_MIN_TRAIN_HOURS_REQUIRED = 49.0
MINI_CASABLANCA_MIN_ROWS = 100
MINI_TRAIN_SELECTION_STRATEGY = "longest_first_with_casablanca_seed"
MINI_PRECACHE_BEFORE_TRAIN = True
MINI_CACHE_WAIT_POLL_SECONDS = 30

MIX_TRAIN_SOURCES = True
SOURCE_SAMPLING_STRATEGY = "round_robin"  # or "temperature"
SOURCE_TEMPERATURE = 0.7
BALANCE_SOURCES_IN_BATCH = True

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 1
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

CACHE_NUM_WORKERS = 4
TRAIN_DATALOADER_NUM_WORKERS = 4
CACHE_SHARD_SIZE = 250
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

FORCE_REBUILD_MANIFEST = False
FORCE_REBUILD_CACHE = False
RUN_SMOKE_TEST = False
SMOKE_TEST_ONLY = False
RUN_SMALL_REAL_SAMPLE_TEST = False
DEBUG_STRICT = False

USE_BF16 = True
USE_FP16 = False
RESUME_FROM_CHECKPOINT = False

# Text normalization policy: remove dataset artifacts, not Arabic spelling.
NORMALIZE_ARABIC = True
REMOVE_ASR_TAGS = True
REMOVE_TATWEEL = True
REMOVE_DIACRITICS = True
NORMALIZE_ALEF = False
NORMALIZE_YAA = False
NORMALIZE_TAA_MARBUTA = False
REMOVE_PUNCTUATION = False
COLLAPSE_REPEATED_PUNCT = True
MAP_WESTERN_TO_ARABIC_PUNCT = False
DROP_EMPTY_AFTER_NORMALIZATION = True

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "auto"

MAX_OMNILINGUAL_SAMPLES_DEBUG = 5
MAX_CASABLANCA_SAMPLES_DEBUG = 5
MAX_QASR_SEGMENTS_DEBUG = 5

PREDICTION_DIR = RUN_DIR / "eval_predictions"
BASELINE_TEST_PREDICTIONS_PATH = PREDICTION_DIR / "mini_50h_100eval_baseline_test_predictions.jsonl"
BASELINE_TEST_METRICS_PATH = PREDICTION_DIR / "mini_50h_100eval_baseline_test_metrics.json"
FORCE_REGENERATE_BASELINE = False
MAX_BASELINE_TEST_SAMPLES = MINI_MAX_EVAL_SAMPLES
BASELINE_GENERATION_LANGUAGE = "Arabic"
SMOKE_CACHE_DIR = CACHE_DIR / "_smoke"

for p in [RUN_DIR, LOG_DIR, OUTPUT_DIR, BEST_MODEL_DIR, MANIFEST_DIR, CACHE_DIR, PREDICTION_DIR]:
    p.mkdir(parents=True, exist_ok=True)
for split in ["train", "val", "test"]:
    (CACHE_DIR / split).mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / split).mkdir(parents=True, exist_ok=True)
    (SMOKE_CACHE_DIR / split).mkdir(parents=True, exist_ok=True)
(MANIFEST_DIR / "split_assignments").mkdir(parents=True, exist_ok=True)

CONFIG = {k: str(v) if isinstance(v, Path) else v for k, v in globals().items() if k.isupper() and k not in {"CONFIG"}}
print(f"Configured MINI run dir: {RUN_DIR}")
print(f"Notebook path: {NOTEBOOK_PATH}")
print(f"Train cap: {MINI_MAX_TRAIN_HOURS}h; eval/test/baseline cap: {MINI_MAX_EVAL_SAMPLES} samples")
"""
    set_source(nb["cells"][2], config)

    set_source(
        nb["cells"][40],
        """def row_duration_seconds(row: dict) -> float:
    try:
        return float(row.get("duration") or 0.0)
    except Exception:
        return 0.0

def hours(rows: List[dict]) -> float:
    return sum(row_duration_seconds(r) for r in rows) / 3600.0

def stable_row_order(rows: List[dict], salt: str) -> List[dict]:
    return sorted(rows, key=lambda r: stable_hash(f"{salt}|{r.get('uid')}", SPLIT_SEED))

def select_rows_up_to_hours(rows: List[dict], target_hours: float, *, salt: str) -> List[dict]:
    selected = []
    total = 0.0
    for row in stable_row_order(rows, salt):
        dur = row_duration_seconds(row)
        if dur <= 0:
            continue
        if selected and total + dur > target_hours * 3600.0:
            continue
        selected.append(row)
        total += dur
        if total >= target_hours * 3600.0:
            break
    return selected

def duration_desc_order(rows: List[dict], salt: str) -> List[dict]:
    return sorted(rows, key=lambda r: (-row_duration_seconds(r), stable_hash(f"{salt}|{r.get('uid')}", SPLIT_SEED)))

def duration_asc_order(rows: List[dict], salt: str) -> List[dict]:
    return sorted(rows, key=lambda r: (row_duration_seconds(r), stable_hash(f"{salt}|{r.get('uid')}", SPLIT_SEED)))

def select_train_subset_by_hours(rows: List[dict], target_hours: float) -> List[dict]:
    target_seconds = float(target_hours) * 3600.0
    selected = []
    selected_uids = set()
    total = 0.0

    # Keep a small Casablanca slice so the mini run still covers both train roots,
    # then fill the 50h cap with longest clips to avoid a 10k-row preprocessing run.
    casa_rows = [r for r in stable_row_order(rows, "mini-casablanca-seed") if r.get("source_group") == "casablanca_relevant_arabic"]
    for row in casa_rows[: int(MINI_CASABLANCA_MIN_ROWS)]:
        dur = row_duration_seconds(row)
        if dur <= 0 or total + dur > target_seconds:
            continue
        selected.append(row)
        selected_uids.add(row["uid"])
        total += dur

    for row in duration_desc_order([r for r in rows if r["uid"] not in selected_uids], "mini-longest"):
        dur = row_duration_seconds(row)
        if dur <= 0:
            continue
        if total + dur > target_seconds:
            continue
        selected.append(row)
        selected_uids.add(row["uid"])
        total += dur
        if total >= target_seconds:
            break

    # Fill small gaps with short rows without exceeding 50h.
    for row in duration_asc_order([r for r in rows if r["uid"] not in selected_uids], "mini-topoff-short"):
        dur = row_duration_seconds(row)
        if dur <= 0:
            continue
        if total + dur > target_seconds:
            continue
        selected.append(row)
        selected_uids.add(row["uid"])
        total += dur
        if total >= target_seconds:
            break

    return stable_row_order(selected, "mini-train-final-2")

def select_eval_rows(rows: List[dict], max_samples: int, *, salt: str) -> List[dict]:
    return stable_row_order(rows, salt)[: int(max_samples)]

ALL_ROWS = build_or_load_full_manifest()
FULL_TRAIN_ROWS = [r for r in ALL_ROWS if r.get("split") == "train"]
FULL_VAL_ROWS = [r for r in ALL_ROWS if r.get("split") == "val"]
FULL_TEST_ROWS = [r for r in ALL_ROWS if r.get("split") == "test"]

TRAIN_ROWS = select_train_subset_by_hours(FULL_TRAIN_ROWS, MINI_MAX_TRAIN_HOURS)
VAL_ROWS = select_eval_rows(FULL_VAL_ROWS, MINI_MAX_EVAL_SAMPLES, salt="mini-val-100")
TEST_ROWS = select_eval_rows(FULL_TEST_ROWS, MINI_MAX_EVAL_SAMPLES, salt="mini-test-100")
ROWS_BY_SPLIT = {"train": TRAIN_ROWS, "val": VAL_ROWS, "test": TEST_ROWS}

selection_summary = {
    "run_id": MINI_RUN_ID,
    "full_counts": {"train": len(FULL_TRAIN_ROWS), "val": len(FULL_VAL_ROWS), "test": len(FULL_TEST_ROWS)},
    "selected_counts": {k: len(v) for k, v in ROWS_BY_SPLIT.items()},
    "selected_hours": {k: hours(v) for k, v in ROWS_BY_SPLIT.items()},
    "train_hours_cap": MINI_MAX_TRAIN_HOURS,
    "eval_sample_cap": MINI_MAX_EVAL_SAMPLES,
    "train_by_source_group": {
        g: {"rows": len(xs), "hours": hours(xs)}
        for g, xs in sorted(collections.defaultdict(list, {g: [r for r in TRAIN_ROWS if r.get("source_group") == g] for g in sorted({r.get("source_group") for r in TRAIN_ROWS})}).items())
    },
}
print(json.dumps(selection_summary, ensure_ascii=False, indent=2, default=str))
(MANIFEST_DIR / "mini_selection_summary.json").write_text(json.dumps(selection_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
jsonl_write(MANIFEST_DIR / "train" / "manifest_train_mini_50h.jsonl", TRAIN_ROWS)
jsonl_write(MANIFEST_DIR / "val" / "manifest_val_mini_100.jsonl", VAL_ROWS)
jsonl_write(MANIFEST_DIR / "test" / "manifest_test_mini_100.jsonl", TEST_ROWS)

assert TRAIN_ROWS, "No training rows found. Check TRAIN_SOURCE_DIRS and manifest parser logs."
assert VAL_ROWS and TEST_ROWS, "Validation/test rows are required from held-out roots."
assert hours(TRAIN_ROWS) >= MINI_MIN_TRAIN_HOURS_REQUIRED, f"Mini train subset is only {hours(TRAIN_ROWS):.2f}h; requested about {MINI_MAX_TRAIN_HOURS:.2f}h."
assert len(VAL_ROWS) <= MINI_MAX_EVAL_SAMPLES and len(TEST_ROWS) <= MINI_MAX_EVAL_SAMPLES
""",
    )

    set_source(
        nb["cells"][42],
        """baseline_test_metrics = run_resumable_test_predictions(
    TEST_ROWS,
    BASELINE_TEST_PREDICTIONS_PATH,
    BASELINE_TEST_METRICS_PATH,
    disable_adapters=True,
    force=FORCE_REGENERATE_BASELINE,
    max_samples=MAX_BASELINE_TEST_SAMPLES,
    label="mini_50h_100eval_baseline_test",
)
print("Mini baseline test metrics saved to", BASELINE_TEST_METRICS_PATH)
print("Mini baseline test predictions saved to", BASELINE_TEST_PREDICTIONS_PATH)
""",
    )

    set_source(
        nb["cells"][44],
        """train_dataset = QwenManifestDataset(TRAIN_ROWS, processor, "train", CACHE_DIR)
eval_dataset = QwenManifestDataset(VAL_ROWS, processor, "val", CACHE_DIR)
test_dataset = QwenManifestDataset(TEST_ROWS, processor, "test", CACHE_DIR)
print("Datasets ready:", len(train_dataset), len(eval_dataset), len(test_dataset))
print("Initial cache indexes:", {s: len(load_cache_index(s, CACHE_DIR)) for s in ["train", "val", "test"]})

# Start this after dataset initialization so cache-index bootstrap cannot race on the same .tmp file.
cache_builder = BackgroundCacheBuilder(ROWS_BY_SPLIT, processor, CACHE_DIR, CACHE_SHARD_SIZE, CACHE_NUM_WORKERS).start()
""",
    )

    set_source(
        nb["cells"][48],
        """def choose_eval_save_steps_after_training() -> int:
    steps_per_epoch = len(BalancedSourceBatchSampler(train_dataset.rows, TRAIN_BATCH_SIZE, SPLIT_SEED))
    steps = max(steps_per_epoch + 1, 200)
    print({"steps_per_epoch": steps_per_epoch, "EVAL_SAVE_STEPS": steps, "note": "set beyond one mini epoch to avoid extra in-training evals"})
    LOGGERS["train"].info("mini eval/save steps set beyond epoch steps_per_epoch=%s steps=%s", steps_per_epoch, steps)
    return steps

if MINI_PRECACHE_BEFORE_TRAIN:
    print("Waiting for mini cache builder to finish before training...")
    while not cache_builder.done.wait(timeout=MINI_CACHE_WAIT_POLL_SECONDS):
        counts = {s: len(load_cache_index(s, CACHE_DIR)) for s in ["train", "val", "test"]}
        print("cache progress", counts)
    if cache_builder.errors:
        raise RuntimeError(f"Background cache builder failed: {cache_builder.errors!r}")
    for ds in [train_dataset, eval_dataset, test_dataset]:
        ds.refresh_index(force=True)
    print("cache ready", {s: len(load_cache_index(s, CACHE_DIR)) for s in ["train", "val", "test"]})

EVAL_SAVE_STEPS = choose_eval_save_steps_after_training()
training_args = make_training_args(eval_save_steps=EVAL_SAVE_STEPS)
trainer = CastFloatInputsQwenTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    tokenizer=processor.tokenizer,
    callbacks=[TimedEvalSaveCallback(10**9), BestConfigCallback(BEST_MODEL_DIR)],
)
print("Recreated trainer with final eval_steps/save_steps:", EVAL_SAVE_STEPS)
""",
    )

    set_source(
        nb["cells"][52],
        """eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
print("mini eval metrics", eval_metrics)
LOGGERS["eval"].info("mini_eval_metrics=%s", eval_metrics)
preview_metrics = run_generation_eval(VAL_ROWS, max_samples=MINI_MAX_EVAL_SAMPLES, name=f"mini_50h_100eval_val_predictions_step_{trainer.state.global_step}")
print("mini validation prediction metrics", preview_metrics)
""",
    )

    set_source(
        nb["cells"][54],
        """test_preview_metrics = run_generation_eval(TEST_ROWS, max_samples=MINI_MAX_EVAL_SAMPLES, name=f"mini_50h_100eval_test_predictions_step_{trainer.state.global_step}")
print("mini test prediction metrics", test_preview_metrics)
""",
    )

    integrity_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 29. Mini Run Integrity Report\n"],
    }
    integrity_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """def prediction_file_health(path: Path) -> dict:
    records = jsonl_read(path)
    empty = [r.get("uid") for r in records if not str(r.get("prediction", "")).strip()]
    return {
        "path": str(path),
        "rows": len(records),
        "empty_predictions": len(empty),
        "empty_prediction_uids": empty[:20],
        "hours": sum(float(r.get("duration") or 0.0) for r in records) / 3600.0,
    }

prediction_health = {
    "baseline_test": prediction_file_health(BASELINE_TEST_PREDICTIONS_PATH),
    "val_predictions": prediction_file_health(Path(preview_metrics["prediction_path"])),
    "test_predictions": prediction_file_health(Path(test_preview_metrics["prediction_path"])),
}

mini_results = {
    "run_id": MINI_RUN_ID,
    "notebook_path": str(NOTEBOOK_PATH),
    "run_dir": str(RUN_DIR),
    "cache_dir": str(CACHE_DIR),
    "selection_summary": selection_summary,
    "baseline_test_metrics": baseline_test_metrics,
    "trainer_eval_metrics": eval_metrics,
    "val_prediction_metrics": preview_metrics,
    "test_prediction_metrics": test_preview_metrics,
    "prediction_health": prediction_health,
    "final_adapter_dir": str(FINAL_DIR),
    "summary_report": str(summary_path),
}
mini_results_path = RUN_DIR / "mini_50h_100eval_results.json"
mini_results_path.write_text(json.dumps(mini_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(mini_results, ensure_ascii=False, indent=2, default=str))

bad = {name: h for name, h in prediction_health.items() if h["empty_predictions"] or h["rows"] == 0}
assert not bad, f"Empty or missing predictions detected: {json.dumps(bad, ensure_ascii=False, default=str)}"
assert prediction_health["baseline_test"]["rows"] <= MINI_MAX_EVAL_SAMPLES
assert prediction_health["val_predictions"]["rows"] <= MINI_MAX_EVAL_SAMPLES
assert prediction_health["test_predictions"]["rows"] <= MINI_MAX_EVAL_SAMPLES
print("MINI RUN INTEGRITY CHECK PASSED:", mini_results_path)
""".splitlines(keepends=True),
    }

    nb["cells"].extend([integrity_cell, integrity_code])
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
