# Safety Doesn't Translate

Lexsi AI Safety Hackathon, **Track 2a #4: Multilingual and code-mixed safety.**

Tiny Aya refuses harmful requests in English. Does it still refuse in Hindi, Marathi
and Hinglish? We measure the gap, find the refusal direction inside the model, and
close the gap with conditional activation steering (no retraining).

**Pipeline (4 Lexsi repos):** CuratorKIT → AuditKIT → SafeTune + CircuitKIT → AuditKIT
**Models:** `CohereLabs/tiny-aya-fire` (South Asia focus, main), `CohereLabs/tiny-aya-global` (comparison)

## Phases

| Phase | Script | Status |
|---|---|---|
| 0. Setup and smoke test | `phase0_smoke_test.py` | ready |
| 1. Build the benchmark (CuratorKIT) | `phase1_build_dataset.py` | ready |
| 2. Baseline attacks and evaluation (AuditKIT, SafeTune evaluate) | `phase2_baseline_eval.py` | next |
| 3. Find the refusal direction per language (SafeTune interpret, CircuitKIT) | `phase3_interp.py` | |
| 4. Conditional steering (SafeTune steer) | `phase4_steer.py` | |
| 5. Final evaluation, ROI, HF artifacts, Space, Article | `phase5_report.py` | |

Data splits are by seed prompt: `direction` (20%, for extracting the direction),
`val` (20%, for tuning α and the threshold), `test` (60%, reported numbers only).

## Setup (Kaggle T4 or Colab)

1. On Hugging Face, accept the terms on these gated pages:
   `CohereLabs/tiny-aya-fire`, `CohereLabs/tiny-aya-global`,
   `walledai/HarmBench`, `walledai/XSTest`.
2. Run:
   ```bash
   pip install -r requirements.txt
   huggingface-cli login
   export HF_USER=<your-hf-username>
   python phase0_smoke_test.py
   python phase1_build_dataset.py
   ```

SafeTune needs Python 3.12+ and CuratorKIT needs 3.11+. Kaggle's current image meets both.

## Ethics
Only prompts are published, never harmful completions. Licences: Tiny Aya is CC-BY-NC.
