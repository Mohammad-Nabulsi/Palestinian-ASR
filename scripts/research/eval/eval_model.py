# Promoted from the research scratch dir (~/eval_model.py) on the 5090 node.
# Score a path-mode parquet with base Whisper, a LoRA adapter, or Cohere.
# See docs/HANDOFF.md for what it produced.
"""Score a path-mode parquet with base Whisper, our LoRA adapter, or Cohere.

Same decode settings as the training-time eval so numbers are comparable.
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.home() / "Palestinian-ASR"))
from scripts.train_whisper_medium_lora_progressive import ParquetAudioTextDataset, make_collate_fn
from scripts.zero_shot_eval.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", type=Path, required=True)
ap.add_argument("--audio-dir", type=Path, required=True)
ap.add_argument("--model", choices=("whisper-base", "whisper-lora", "cohere"), required=True)
ap.add_argument("--adapter", type=Path, default=None)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--tag", default="")
a = ap.parse_args()

ds = ParquetAudioTextDataset(a.parquet, audio_dir=a.audio_dir)
print(f"[{a.tag}] {a.parquet.name}: {len(ds)} rows", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
t0 = time.time()

if a.model == "cohere":
    # Use the repo's own predictor: real processor->generate->decode path,
    # splits audio past the processor's 35s ceiling and reassembles it, and
    # halves the batch on CUDA OOM instead of dying.
    from scripts.zero_shot_eval.cohere_predict import CoherePredictor, CoherePredictConfig
    pred = CoherePredictor(CoherePredictConfig())
    refs, hyps = [], []
    for i in range(0, len(ds), a.batch_size):
        batch = [ds[j] for j in range(i, min(i + a.batch_size, len(ds)))]
        hyps += pred.predict([b["audio"] for b in batch])
        refs += [b["text"] for b in batch]
        if (i // a.batch_size) % 10 == 0:
            print(f"[{a.tag}] {i}/{len(ds)} ({time.time()-t0:.0f}s)", flush=True)
else:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    BASE = "openai/whisper-medium"
    proc = WhisperProcessor.from_pretrained(BASE, language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=dt)
    model.generation_config.language = "arabic"; model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if a.model == "whisper-lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.adapter))
        print(f"[{a.tag}] loaded adapter {a.adapter}", flush=True)
    model = model.to(dev).eval()
    collate = make_collate_fn(proc)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    refs, hyps = [], []
    for bi, b in enumerate(loader):
        feats = b["input_features"].to(device=dev, dtype=model.dtype)
        with torch.no_grad():
            gen = model.generate(input_features=feats, max_new_tokens=256,
                                 no_repeat_ngram_size=3, repetition_penalty=1.2,
                                 language="arabic", task="transcribe")
        hyps += [t.strip() for t in proc.tokenizer.batch_decode(gen, skip_special_tokens=True)]
        refs += b["texts"]
        if bi % 10 == 0:
            print(f"[{a.tag}] batch {bi+1}/{len(loader)} ({time.time()-t0:.0f}s)", flush=True)

m = evaluate(refs, hyps)
m_small = {k: v for k, v in m.items() if k != "per_utterance"}
m_small.update({"model": a.model, "adapter": str(a.adapter) if a.adapter else None,
                "parquet": str(a.parquet), "elapsed_s": time.time() - t0})
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(m, ensure_ascii=False, indent=2))
print(f"RESULT [{a.tag}] {a.model}: n={m_small['n_scored']} WER={m_small['wer']:.4f} CER={m_small['cer']:.4f}", flush=True)
