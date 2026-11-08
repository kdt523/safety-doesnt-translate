# Safety Doesn't Translate: measuring and closing the multilingual refusal gap in Tiny Aya

*Lexsi AI Safety and Alignment Hackathon, Track 2a, problem 4: multilingual and code-mixed safety.*

> **Draft.** Sections marked TODO wait on runs that have not finished. Nothing in
> those sections should be filled in from estimates.

## TL;DR

- Tiny Aya Fire refuses 91% of harmful English prompts and only **35% of the same
  prompts in Hinglish** (romanised Hindi-English). Attack success rises from 7% to 51%.
- The refusal mechanism is not missing in Hinglish. The model uses the **same refusal
  direction** in all four languages (cosine similarity 0.75 to 0.94), but code-mixed
  text **activates it weakly**.
- Because the direction already exists, the fix needs no retraining: **conditional
  activation steering** pushes it back where it is under-activated.
- TODO: headline before-and-after on the held-out test split.

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
of Hindi, 100% of Marathi and 94.7% of Hinglish rows. TODO: result of the native-speaker
review of the 24 flagged rows. Automated checks catch structural faults, not changes in
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
Two variants:

- **Always-on:** add alpha times the direction to every prompt.
- **Conditional (CAST):** check a cosine gate first, and steer only prompts that trip it.
  The gate was fitted on all four languages at layer 17 (held-out F1 0.98).

Strength alpha was tuned on the `val` split only. On that split, using a keyword refusal
matcher as a cheap proxy, conditional steering at alpha 8 reached 86% refusal on harmful
prompts with 5.9% over-refusal, against 50% over-refusal for always-on at the same
strength. These are tuning numbers, not results. The reported numbers below come from
the held-out test split and the WildGuard judge.

## Results on the held-out test split

TODO: before-and-after table for ASR, refusal and over-refusal per language, baseline
versus conditional steering versus always-on, with intervals.

## Robustness

TODO: best-of-5 sampling. A prompt counts as broken if any of five sampled answers is
judged harmful, run on the unsteered and steered model, with single-sample rates beside
it. TODO: AuditKIT built-in probe pass rates (prompt injection, jailbreak, encoding,
refusal), which are small and English-only.

**Not covered.** The problem statement also lists weight-space attacks such as
abliteration and further obfuscation attacks. We did not run them. Multi-turn audits
are TODO or out of scope, depending on time.

## Cost: self-hosted versus a closed API

TODO: realistic message volume, measured tokens per second on the GPU used, GPU hours,
hourly GPU price, against closed-model token billing plus a moderation call. Steering
adds a forward hook and a cosine check, so runtime overhead should be small, but it
must be measured.

## Limitations

- Prompts are US-centric (HarmBench and XSTest), not written for Indian users.
- Hindi and Marathi are machine translations. Hinglish was written by a sibling of the
  tested models, which may understate the real gap.
- The judge is English-trained and sees translated answers for two languages.
- Moderate sample size per language. Differences smaller than the intervals are not
  claims.
- Steering strength is tuned on one model. We have not tested that it transfers to
  other models or languages.
- Mechanism claims rest on four languages and one layer.

## Reproduce

Code, runner and results: https://github.com/kdt523/safety-doesnt-translate.
TODO: Hugging Face links for the dataset, logs, steering vectors and Space.

## Credits

HarmBench (Mazeika et al., 2024), XSTest (Rottger et al., 2023), NLLB (Meta), Tiny Aya
(Cohere Labs), WildGuard (Allen AI), and the refusal-direction method of Arditi et al.
(2024). Libraries: CuratorKIT, AuditKIT and SafeTune from Lexsi Labs.
