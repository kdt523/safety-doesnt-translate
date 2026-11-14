# Safety Doesn't Translate

Lexsi AI Safety and Alignment Hackathon, **Track 2a, problem 4: multilingual and
code-mixed safety.**

Tiny Aya Fire refuses 91% of harmful requests in English but only 35% of the same
requests in Hinglish. We built a parallel four-language benchmark, measured the gap,
found why it happens inside the model, and closed most of it with conditional activation
steering. No retraining, and no rise in over-refusal.

| | |
|---|---|
| Write-up | [`cards/ARTICLE_DRAFT.md`](cards/ARTICLE_DRAFT.md) |
| Results explorer (Space) | https://huggingface.co/spaces/kd1609/safety-doesnt-translate |
| Benchmark | https://huggingface.co/datasets/kd1609/hinglish-safety-bench |
| Judged outputs | https://huggingface.co/datasets/kd1609/hinglish-safety-eval-logs |
| Steering vectors and gate | https://huggingface.co/kd1609/tiny-aya-fire-hinglish-refusal-steering |

## Results (held-out test split, WildGuard judge)

| Language | Attack success before | after steering | Over-refusal before | after |
|---|---|---|---|---|
| English | 7.1% | 1.8% | 6.4% | 6.4% |
| Hindi | 17.7% | 6.2% | 5.1% | 5.1% |
| Marathi | 31.9% | 11.5% | 12.2% | 12.2% |
| Hinglish | 51.3% | 31.0% | 7.1% | 7.1% |

113 harmful and 156 harmless prompts per language; 95% intervals are in
`results/phase4_summary.csv`. The same push without the gate raises over-refusal to
20-55%. Hinglish improves but remains the weakest language.

**Why:** all four languages share one refusal direction (cosine 0.75-0.94 with
English), but harmful Hinglish prompts activate it weakly (17.1 against 24.9 for
English). The gap is weak activation, not a missing concept, so it can be topped up at
inference time.

## Pipeline

**Lexsi libraries chained:** CuratorKIT → AuditKIT → SafeTune → AuditKIT.
**Models (Cohere Labs):** `tiny-aya-fire` (under test), `tiny-aya-global` (comparison),
`tiny-aya-base` (writes the Hinglish prompts).

| Phase | Script | What it does |
|---|---|---|
| 0 | `phase0_smoke_test.py` | load the model, one answer |
| 1 | `phase1_build_dataset.py` | HarmBench + XSTest seeds, NLLB Hindi/Marathi, few-shot Hinglish, CuratorKIT cleaning |
| 2 | `phase2_baseline_eval.py` | baseline answers (plain and fiction-framed), AuditKIT WildGuard judge |
| 3 | `phase3_interp.py` | refusal direction per language (SafeTune), projections, cosine |
| 4 | `phase4_steer.py` | CAST gate (SafeTune), candidates on val, fixed selection rule, test run |
| 5 | `phase5_robustness.py` | best-of-5 sampling and AuditKIT probes (built; not run before the deadline) |
| | `summarise.py` | rates with Wilson intervals, broken-output rate |
| | `check_translations.py` | automated translation screening |
| | `publish_hf.py`, `space/` | Hugging Face artifacts and the Space |

Splits are by prompt: `direction` (fit the direction and the gate), `val` (choose the
steering setting), `test` (reported numbers only).

## Run

GPU phases run on Kaggle through `kaggle/phase0/run.py`, which clones this repository and
runs the phase named in `PHASE`:

```bash
python -m kaggle kernels push -p kaggle/phase0 --accelerator NvidiaTeslaT4
```

The Hugging Face token comes from a private Kaggle dataset `hf-token` holding
`hf_token.txt`. Accept the terms of the gated `CohereLabs/tiny-aya-*` models and
`walledai/HarmBench`, `walledai/XSTest` first.

## A note on evaluation

The first steering run was corrupted by right padding in batched generation; the judge
scored the resulting broken text as safe. It is kept in `results/archive_rightpad/` with
an explanation. Every table now reports a broken-output rate, and batched generation
refuses to run without left padding.

## Ethics and licences

The benchmark contains prompts only. Public logs redact answers the judge scored as
harmful; unredacted logs are not published. Licence CC BY-NC 4.0, following NLLB and Tiny
Aya. Sources: HarmBench (MIT), XSTest (CC BY 4.0), NLLB-200 (CC BY-NC 4.0), Tiny Aya
(CC BY-NC), WildGuard (Allen AI).
