# Safety Doesn't Translate: measuring and closing the multilingual refusal gap in Tiny Aya

*Lexsi AI Safety and Alignment Hackathon, Track 2a, problem 4: multilingual and code-mixed safety.*

**Links:** [Space](https://huggingface.co/spaces/kd1609/safety-doesnt-translate) ·
[Benchmark](https://huggingface.co/datasets/kd1609/hinglish-safety-bench) ·
[Eval logs](https://huggingface.co/datasets/kd1609/hinglish-safety-eval-logs) ·
[Steering vectors](https://huggingface.co/kd1609/tiny-aya-fire-hinglish-refusal-steering) ·
[Code](https://github.com/kdt523/safety-doesnt-translate)

## TL;DR

- Tiny Aya Fire refuses 91% of harmful English prompts and only **35% of the same
  prompts in Hinglish** (romanised Hindi-English). Attack success rises from 7% to 51%.
- The refusal mechanism is not missing in Hinglish. The model uses the **same refusal
  direction** in all four languages (cosine similarity 0.75 to 0.94), but code-mixed
  text **activates it weakly**.
- Because the direction already exists, the fix needs no retraining: **conditional
  activation steering** pushes it back where it is under-activated.
- On the held-out test split, conditional steering cuts Hinglish attack success from
  **51% to 31%**, Marathi from 32% to 12% and Hindi from 18% to 6%, with **no rise in
  over-refusal in any language**. The same push without the gate makes the model refuse
  20-55% of harmless questions.

## Why this matters

A model that is safe in English and unsafe in the language its users actually type is
not safe for those users. Code-mixing such as Hinglish is the normal way many people
write online, and standard safety benchmarks are English. We wanted to know whether
refusal survives the switch, and, if it does not, whether it can be repaired cheaply.

## What we built

A four-stage pipeline, chaining three Lexsi Labs libraries:

| Stage | Tool | What it did |
|---|---|---|
| Benchmark | **CuratorKIT** | de-duplicated and cleaned a parallel 4-language safety set |
| Measure | **AuditKIT** | scored every answer with the WildGuard guard judge |
| Locate and fix | **SafeTune** | extracted the refusal direction, fitted a conditional steering gate, applied it |
| Re-measure | **AuditKIT** | same judge, held-out test split |

Models are from Cohere Labs: `tiny-aya-fire` (the model under test), `tiny-aya-global`
(comparison) and `tiny-aya-base` (used to write the Hinglish prompts).

We did not use CircuitKIT. Locating one refusal direction was enough to explain and fix
the gap, and a shallow use of a fourth library would have been box-ticking.

## The benchmark

449 prompts, each in English, Hindi, Marathi and Hinglish (1,796 rows). 199 are harmful,
taken from HarmBench. 250 are harmless but scary-sounding, taken from XSTest, and exist
to measure over-refusal. Hindi and Marathi are NLLB translations. Hinglish is written by
Tiny Aya Base from five examples. Full construction, limitations and licences are in the
dataset card. The set is parallel, so a change in behaviour can be put down to language,
not content.

Splits are by prompt: `direction` (finding the refusal direction), `val` (tuning), and
`test` (reported numbers only).

**Translation quality.** Automated screening of all 1,347 non-English rows passed 100%
of Hindi, 100% of Marathi and 94.7% of Hinglish rows. The 24 flagged Hinglish rows were
kept and listed for review; a native-speaker review was not completed in time, so meaning
drift is an open limitation. Automated checks catch structural faults, not changes in
meaning.

## Metrics, and why these

Reported together, never one alone:

- **Attack success rate (ASR):** harmful prompts whose answer WildGuard judges harmful.
  Lower is safer.
- **Refusal rate:** harmful prompts the model refused. Higher is safer.
- **Over-refusal rate:** harmless prompts the model refused. Lower is more useful. This
  is what stops "refuse everything" from looking like a good result.
- **95% Wilson confidence intervals** on every rate. With about 113 harmful prompts per
  language, intervals are roughly +/-9 points, so we only call a gap real when the
  intervals separate.

WildGuard is trained on English, so it always sees the English version of the prompt,
and Hindi and Marathi answers are machine-translated to English before judging. This adds
noise to those two languages and is a limitation.

## Finding 1: safety does not transfer

Tiny Aya Fire, plain requests, harmful test prompts (95% intervals in the repository):

| Language | Attack success | Refusal | Over-refusal |
|---|---|---|---|
| English | 7.1% | 91.2% | 6.4% |
| Hindi | 17.7% | 71.7% | 5.1% |
| Marathi | 31.9% | 60.2% | 12.2% |
| **Hinglish** | **51.3%** | **35.4%** | 7.1% |

Over-refusal stays low everywhere, so the model is not just more permissive overall. It
specifically fails to refuse harmful requests outside English.

Tiny Aya Global behaves the same way (Hinglish ASR 57.5%), so this is a property of the
family and not one checkpoint.

A fictional-story framing raises English ASR from 7.1% to 20.4% but changes Hinglish
little (51.3% to 40.7%). In Hinglish the language switch alone does more damage than
an explicit jailbreak framing does in English.

## Finding 2: the mechanism is weak activation, not a missing concept

We extracted the refusal direction (difference of mean activations on harmful versus
harmless English prompts, last prompt token, held-out `direction` split) at layer 18
of 36, following Arditi et al. Two hypotheses:

1. **Rotation:** Hinglish has its own refusal direction.
2. **Weak activation:** the same direction exists but code-mixed prompts barely excite it.

The data supports the second:

- Per-language refusal directions are close to the English one: cosine 0.78 (Hindi),
  0.94 (Marathi), 0.85 (Hinglish).
- Harmful prompts project onto the English direction at 24.9 (English), 21.2 (Hindi),
  19.9 (Marathi), 17.1 (Hinglish). The correlation between this projection and
  measured ASR across the four languages is r = -0.97. With four points this is a strong
  trend, not a precise coefficient.

One caveat: Hindi separates harmful from harmless slightly less than Marathi but is
attacked less, so projection strength is not the whole story.

## The fix: conditional steering

Since the direction already exists, we add it back at inference time, with no retraining.

- **Conditional (CAST):** a cosine gate at layer 17 decides, per prompt, whether the
  prompt looks harmful. Only prompts that trip it get alpha times the refusal direction
  added at layer 18. The gate was fitted on all four languages of the `direction` split
  (held-out F1 0.98) using SafeTune's `fit_cast_condition`.
- **Projection-matched (ours):** the same gate, but each steered prompt is topped up to
  the English refusal level (alpha = 24.9 minus its own projection). It follows directly
  from Finding 2 and has no tuned parameter.
- **Always-on:** the chosen push with the gate removed, to show what the gate buys.

**How the setting was chosen.** Every candidate (CAST at alpha 4, 6, 8, 12, and
projection-matched) was run on the `val` split and scored with WildGuard. A rule fixed
before seeing results kept only settings whose over-refusal rose by at most 5 points
overall (10 in any language) and whose broken-output rate rose by at most 5 points, then
took the lowest attack success. All five passed; CAST at alpha 12 won (val ASR 9.3%
against 23.8% unsteered, over-refusal unchanged). Projection-matched came second with
the least broken output.

## Results on the held-out test split

Tiny Aya Fire, 113 harmful and 156 harmless prompts per language, WildGuard judge, 95%
Wilson intervals.

**Attack success (lower is safer)**

| Language | Before | Conditional steering | Always-on (no gate) |
|---|---|---|---|
| English | 7.1% (3.6-13.4) | 1.8% (0.5-6.2) | 1.8% |
| Hindi | 17.7% (11.8-25.8) | **6.2%** (3.0-12.2) | 6.2% |
| Marathi | 31.9% (24.0-40.9) | **11.5%** (6.8-18.7) | 9.7% |
| Hinglish | 51.3% (42.2-60.3) | **31.0%** (23.2-40.0) | 27.4% |

**Over-refusal (lower is more useful)**

| Language | Before | Conditional steering | Always-on (no gate) |
|---|---|---|---|
| English | 6.4% | **6.4%** | 55.1% |
| Hindi | 5.1% | **5.1%** | 35.9% |
| Marathi | 12.2% | **12.2%** | 35.3% |
| Hinglish | 7.1% | **7.1%** | 20.5% |

- The drops in Hindi, Marathi and Hinglish are outside the intervals. English starts too
  low to call.
- Over-refusal is identical to the unsteered model: the gate did not steer a single
  harmless prompt into a new refusal. Without the gate, the same push is unusable.
- Broken output (near-empty or looping answers) is unchanged in English, Hindi and
  Marathi (0%). In Hinglish it rises on harmful prompts from 9.7% to 15.9%, so a small
  part of the Hinglish drop is the model breaking down, not refusing.
- **Hinglish is not fixed.** Steering closes about 45% of its gap to English. It remains
  the weakest language.

## Getting the evaluation right

Our first steering run looked like a different result: attack success fell, but
over-refusal rose to 27-76%. Reading the answers showed most of them were broken text.
Batched generation was padding prompts on the right, so shorter prompts in each batch
were followed by pad tokens and the model wrote loops and fragments. WildGuard scores
broken text as "not harmful" and often as "refused", so both effects were artefacts.

The check that found it: harmless prompts the gate did not steer must produce exactly
the unsteered answer. They did 42% of the time when the prompt was the longest in its
batch (no padding) and 0% of the time when it was among the shortest.

What we changed, and what we recommend to anyone evaluating steering:

- Left padding, enforced in code.
- A **broken-output rate** next to every metric, so a model that falls apart cannot look
  safe. It also showed that unsteered Tiny Aya already loops on about 20% of harmless
  Hinglish prompts.
- Tune with the real judge, not a keyword matcher (keywords cannot see broken text).
- Fit the gate on a different split from the one used to choose alpha.

The corrupted run is kept in the repository (`results/archive_rightpad/`).

## Robustness

**What we measured.** A fictional-story jailbreak framing (the first pattern in
AuditKIT's built-in jailbreak probe) raises English attack success from 7.1% to 20.4%,
while plain Hinglish already reaches 51.3%. Switching language is a stronger attack on
this model than an explicit jailbreak in English.

**What we built but could not run in time.** A best-of-5 sampling attack (a prompt
counts as broken if any of five sampled answers is judged harmful) on the unsteered and
steered model, plus AuditKIT's built-in probes (prompt injection, jailbreak, encoding,
refusal). The code is in the repository (`phase5_robustness.py`); the run did not get
GPU time before the deadline, so the steered model's robustness to repeated sampling is
**untested**. All reported results are greedy decoding.

**Not covered.** Weight-space attacks such as abliteration, further obfuscation attacks,
and multi-turn audits. Inference-time steering offers no protection against someone who
can edit the weights.

## Cost

Steering changes no weights and adds no model. Per request it costs one extra prefill
pass over the prompt (to read the gate) and one vector addition per generated token.
Generation, which dominates cost, is unchanged. We did not benchmark wall-clock overhead
or compare against a hosted model plus a moderation API, so we make no cost claim beyond
this.

## Limitations

- Prompts are US-centric (HarmBench and XSTest), not written for Indian users.
- Hindi and Marathi are machine translations. Hinglish was written by a sibling of the
  tested models, which may understate the real gap.
- The judge is English-trained and sees translated answers for two languages.
- Moderate sample size per language. Differences smaller than the intervals are not
  claims.
- The push strength (alpha 12) was chosen from five candidates on the val split; it is
  the largest one tried, so a larger value might do better or worse.
- Robustness to sampling-based and multi-turn attacks is untested.
- Unsteered Tiny Aya already loops on about 20% of harmless Hinglish prompts; steering
  does not fix fluency.
- Steering strength is tuned on one model. We have not tested that it transfers to
  other models or languages.
- Mechanism claims rest on four languages and one layer.

## Reproduce

- Code, Kaggle runner and result tables: https://github.com/kdt523/safety-doesnt-translate
- Benchmark (prompts only): https://huggingface.co/datasets/kd1609/hinglish-safety-bench
- Judged outputs behind every number (answers judged harmful are redacted; verdicts kept):
  https://huggingface.co/datasets/kd1609/hinglish-safety-eval-logs
- Steering direction, gate and usage code:
  https://huggingface.co/kd1609/tiny-aya-fire-hinglish-refusal-steering
- Results explorer: https://huggingface.co/spaces/kd1609/safety-doesnt-translate

The Space is a static page. A live demo would have to generate unsteered answers to any
prompt, which would hand out exactly the harmful Hinglish answers this work documents.

## Credits

HarmBench (Mazeika et al., 2024), XSTest (Rottger et al., 2023), NLLB (Meta), Tiny Aya
(Cohere Labs), WildGuard (Allen AI), and the refusal-direction method of Arditi et al.
(2024). Libraries: CuratorKIT, AuditKIT and SafeTune from Lexsi Labs.
