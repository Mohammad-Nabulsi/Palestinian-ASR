#!/usr/bin/env python
"""
Validate the REAL, FIXED training path on CPU against omniASR_LLM_300M:
manual-LoRA apply_lora (fairseq2.nn.Linear can't be wrapped by PEFT) -> forward loss ->
backward -> optimizer step -> checkpoint save -> reload. One real apc clip.
GPU-only bits (bf16 autocast, 8-bit Adam) are substituted with fp32 AdamW on CPU.
"""
import os, json, time
from pathlib import Path
os.environ["ASR_ENV_ROOT"] = "/workspace/asr_env"
os.environ.setdefault("HF_HOME", "/workspace/asr_env/models/hf")
os.environ["WANDB_MODE"] = "disabled"; os.environ["WANDB_DISABLED"] = "true"

NB = json.loads(Path("/root/asr/Palestinian-ASR/asr_model_agnostic_finetune.ipynb").read_text())
cells = {c.get("id"): "".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"}
G = {"__name__": "trn"}
for cid in ["5714e106", "628fd594", "b5a6fb26", "267f4147", "312a0a67", "12e6305e", "27065793"]:
    src = "\n".join(l for l in cells[cid].splitlines() if not l.lstrip().startswith("!"))
    exec(compile(src, f"<cell {cid}>", "exec"), G)

import numpy as np, torch
get_adapter = G["get_adapter"]; ConfigAPI = G["ConfigAPI"]; DEVICE = G["DEVICE"]

fx = np.load("/workspace/asr_env/g5_fixture/audio.npz")
meta = json.load(open("/workspace/asr_env/g5_fixture/meta.json"))
sample = {"audio": {"array": fx["a0"], "sampling_rate": 16000}, "text": meta[0]["text"]}  # 4.29s

name = "omnilingual-asr/omniASR_LLM_300M"
adapter = get_adapter(name, lang="arb_Arab"); adapter.load_base()

print("\n=== apply_lora (manual injection into fairseq2 Linear) ===")
lora = ConfigAPI.lora(name)
adapter.apply_lora(lora)
params = [p for p in adapter.model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in params)
lora_keys = [n for n, p in adapter.model.named_parameters() if p.requires_grad]
print(f"trainable params={n_trainable} across {len(lora_keys)} LoRA tensors (>0 ? {n_trainable>0})")
assert n_trainable > 0 and all("lora_" in k for k in lora_keys)

print("\n=== 3 real train steps (forward loss -> backward -> AdamW step) ===")
b = adapter.collate([adapter.preprocess(sample)])
b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
opt = torch.optim.AdamW(params, lr=5e-3)   # high lr so a 3-step move is visible
adapter.model.train()
losses = []
for step in range(3):
    t0 = time.time()
    loss = adapter.train_step(b)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    losses.append(float(loss))
    print(f"  step {step}: loss={float(loss):.4f} | grad_norm={float(gnorm):.3f} "
          f"| finite={bool(torch.isfinite(loss))} | {time.time()-t0:.1f}s")
assert all(np.isfinite(losses))
print(f"loss trajectory: {[round(l,4) for l in losses]} | moved ? {losses[-1] != losses[0]}")
assert losses[0] != losses[-1], "loss did not change -> LoRA grads not flowing"

print("\n=== checkpoint save + reload (LoRA state_dict, on overlay not /workspace) ===")
best = Path("/tmp/claude-0/-root-asr-Palestinian-ASR/4137da38-bb95-4b90-a13b-20eaed8d4580/scratchpad/ckpt_lora")
best.mkdir(parents=True, exist_ok=True)
sd = {k: v.detach().clone() for k, v in adapter.model.state_dict().items() if "lora_" in k}
torch.save(sd, best / "adapter.pt")
print(f"saved {len(sd)} LoRA tensors -> {best/'adapter.pt'}")
# perturb one trained tensor, reload, confirm exact restore
k0 = lora_keys[0]
with torch.no_grad():
    dict(adapter.model.named_parameters())[k0].add_(1.0)
reload_sd = torch.load(best / "adapter.pt", map_location=DEVICE)
adapter.model.load_state_dict(reload_sd, strict=False)
restored = torch.equal(dict(adapter.model.named_parameters())[k0].detach(), reload_sd[k0])
print(f"reload via load_state_dict(strict=False) | restored perturbed tensor exactly ? {restored}")
assert restored

print("\n" + "="*72)
print(f"TRAIN-PATH CPU VALIDATION PASS | manual-LoRA trainable={n_trainable} | "
      f"loss {losses[0]:.3f}->{losses[-1]:.3f} | save+reload OK")
print("Covers: apply_lora + backward through real fairseq2 model + AdamW step + ckpt round-trip.")
print("NOT covered (GPU only): bf16 autocast, bitsandbytes 8-bit Adam, full multi-epoch loop, 1B.")
print("="*72)
