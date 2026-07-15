#!/usr/bin/env python
"""
No-GPU / no-disk structural test of the REWRITTEN OmniASRAdapter train path.

It mocks fairseq2's Seq2SeqBatch so we can run OmniASRAdapter.collate + train_step and assert
the batch the adapter hands to `model(...)` is shaped/typed the way the real
Wav2Vec2LlamaModel.forward expects (per DISCOVERY.md). This proves MY wiring is right; it does
NOT prove the real model accepts it (that is G1/G3/G4/G5 on the real weights).
"""
import os, sys, json, types
from pathlib import Path
import numpy as np, torch, torch.nn as nn

os.environ["ASR_ENV_ROOT"] = "/tmp/claude-0/-root-asr-Palestinian-ASR/4137da38-bb95-4b90-a13b-20eaed8d4580/scratchpad/asr_env_mock"

# --- mock fairseq2.datasets.batch.Seq2SeqBatch (a plain dataclass-ish recorder) ------------
fake_pkg = types.ModuleType("fairseq2"); fake_pkg.__path__ = []
fake_ds  = types.ModuleType("fairseq2.datasets"); fake_ds.__path__ = []
fake_bt  = types.ModuleType("fairseq2.datasets.batch")
class Seq2SeqBatch:
    def __init__(self, source_seqs, source_seq_lens, target_seqs, target_seq_lens, example):
        self.source_seqs = source_seqs; self.source_seq_lens = source_seq_lens
        self.target_seqs = target_seqs; self.target_seq_lens = target_seq_lens
        self.example = example
fake_bt.Seq2SeqBatch = Seq2SeqBatch
sys.modules["fairseq2"] = fake_pkg
sys.modules["fairseq2.datasets"] = fake_ds
sys.modules["fairseq2.datasets.batch"] = fake_bt

# --- pull the REAL adapter class out of the notebook (cells 1,2,3,5,6) --------------------
NB = json.loads(Path("/root/asr/Palestinian-ASR/asr_model_agnostic_finetune.ipynb").read_text())
cells = {c.get("id"): "".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"}
G = {"__name__": "nb"}
for cid in ["5714e106", "628fd594", "b5a6fb26", "267f4147", "312a0a67", "12e6305e"]:
    src = "\n".join(l for l in cells[cid].splitlines() if not l.lstrip().startswith("!"))
    exec(compile(src, f"<cell {cid}>", "exec"), G)
OmniASRAdapter = G["OmniASRAdapter"]

# --- a fake Wav2Vec2LlamaModel: forward(batch) inspects the batch, returns a scalar loss ---
RECORDED = {}
class FakeLlama(nn.Module):
    def __init__(self): super().__init__(); self.p = nn.Parameter(torch.zeros(1))
    def forward(self, batch):
        RECORDED["type"] = type(batch).__name__
        RECORDED["source_seqs"] = tuple(batch.source_seqs.shape)
        RECORDED["source_dtype"] = batch.source_seqs.dtype
        RECORDED["source_seq_lens"] = batch.source_seq_lens.tolist()
        RECORDED["target_seqs"] = tuple(batch.target_seqs.shape)
        RECORDED["target_dtype"] = batch.target_seqs.dtype
        RECORDED["target_seq_lens"] = batch.target_seq_lens.tolist()
        RECORDED["example"] = batch.example
        return (batch.source_seqs.float().sum() * 0.0 + self.p.sum())  # finite scalar

# --- build adapter WITHOUT load_base (no weights / no package) -----------------------------
ad = OmniASRAdapter("omnilingual-asr/omniASR_LLM_300M", lang="arb_Arab")
ad.name = ad.model_name
ad.model = FakeLlama()
ad.pad_idx = 1
# fake tokenizer encoder: map chars to ids (>=2 so pad_idx=1 is distinct)
ad._encoder = lambda text: torch.tensor([ord(c) % 50 + 2 for c in text], dtype=torch.int64)

# --- run the real collate + train_step -----------------------------------------------------
def ex(text, secs, seed):
    rng = np.random.default_rng(seed)
    return {"audio": {"array": rng.standard_normal(int(secs*16000)).astype(np.float32),
                      "sampling_rate": 16000}, "text": text}

feats = [ad.preprocess(ex(t, s, i)) for i, (t, s) in enumerate([("مرحبا", 0.5), ("اختبار طويل", 1.0)])]
batch = ad.collate(feats)
print("collate keys:", sorted(batch.keys()))
print("collate labels padded with pad_idx? ",
      bool((batch["labels"] == ad.pad_idx).any()) and (-100 not in batch["labels"]))

loss = ad.train_step(batch)
print("\ntrain_step -> loss:", float(loss), "| finite?", bool(torch.isfinite(loss)))
print("batch handed to model.forward:")
for k, v in RECORDED.items():
    print(f"   {k:16} = {v}")

# --- assertions: this is what DISCOVERY.md says the real forward needs --------------------
N = len(feats)
assert RECORDED["type"] == "Seq2SeqBatch"
assert RECORDED["source_seqs"][0] == N and RECORDED["target_seqs"][0] == N
assert RECORDED["target_dtype"] == torch.int64
assert RECORDED["source_seq_lens"] == [int(0.5*16000), int(1.0*16000)]          # true audio lens
assert RECORDED["target_seq_lens"] == [len(feats[0]["labels"]), len(feats[1]["labels"])]  # true label lens
assert RECORDED["example"] == {"lang": ["arb_Arab", "arb_Arab"]}                # lang carried in example
assert -100 not in batch["labels"]                                             # NOT the -100 convention
print("\nPASS: adapter builds a correct Seq2SeqBatch (shapes/dtype/lengths/lang).")
print("NOTE: validates wiring only. Real model acceptance = G1/G3/G4/G5 on GPU-box weights.")
