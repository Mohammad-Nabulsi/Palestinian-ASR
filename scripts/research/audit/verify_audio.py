# Promoted from the research scratch dir (~/verify_audio.py) on the 5090 node.
# Silent/corrupt audio guard over an extracted corpus.
# See docs/HANDOFF.md for what it produced.
import sys, random, collections
from pathlib import Path
import numpy as np, soundfile as sf, pyarrow.parquet as pq

root = Path(sys.argv[1]); adir = Path(sys.argv[2]); n_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 300
random.seed(0); fails = []
for split in ("train", "val", "test"):
    f = root / f"{split}.parquet"
    if not f.exists():
        print(f"{split}: MISSING"); continue
    t = pq.read_table(f)
    cols = t.column_names
    paths = t.column("audio_path").to_pylist()
    durs = t.column("duration").to_pylist()
    srcs = collections.Counter(t.column("source").to_pylist())
    spk = len(set(t.column("speaker_key").to_pylist()))
    hours = sum(durs) / 3600
    print(f"{split:6s} rows={t.num_rows:7d} speakers={spk:5d} hours={hours:8.2f} sources={dict(srcs)}")
    if "audio" in cols: fails.append(f"{split}: embedded audio column present")
    miss = [p for p in paths if not (adir / p).exists()]
    if miss: fails.append(f"{split}: {len(miss)} wavs missing on disk")
    sample = random.sample(paths, min(n_sample, len(paths)))
    peaks, bad = [], []
    for p in sample:
        d, sr = sf.read(adir / p, dtype="int16")
        pk = int(np.abs(d).max()); peaks.append(pk)
        if pk <= 2: bad.append(p)
        if sr != 16000: fails.append(f"{split}: {p} sr={sr}")
    print(f"       sampled {len(sample)} wavs: peak min={min(peaks)} median={int(np.median(peaks))} max={max(peaks)} silent={len(bad)}")
    if bad: fails.append(f"{split}: {len(bad)} silent wavs in sample")
print()
print("AUDIO VERIFY: " + ("ALL PASSED" if not fails else "FAILURES: " + str(fails)))
sys.exit(1 if fails else 0)
