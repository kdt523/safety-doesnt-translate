---
license: cc-by-nc-4.0
language:
- en
- hi
- mr
task_categories:
- text-generation
tags:
- safety
- red-teaming
- multilingual
- code-mixed
- hinglish
- refusal
pretty_name: Hinglish Safety Bench
size_categories:
- 1K<n<10K
---

# Hinglish Safety Bench

A parallel safety benchmark for **English, Hindi, Marathi and Hinglish** (romanised,
code-mixed Hindi-English). Built for the Lexsi AI Safety and Alignment Hackathon,
Track 2a, problem 4: *test whether safety holds across languages and mixed-language
prompts, then close the gaps*.

Every prompt exists in all four languages, so a difference in model behaviour can be
attributed to the language and not to the content.

## What is in it

| | Unique prompts | Rows (x4 languages) |
|---|---|---|
| Harmful (should be refused) | 199 | 796 |
| Harmless but scary-sounding (should be answered) | 250 | 1,000 |
| **Total** | **449** | **1,796** |

The harmless half exists to measure **over-refusal**: a model that refuses everything
looks safe on the harmful half alone.

### Splits

Splits are by prompt, so all four languages of a prompt always land in the same split.

| Split | Harmful | Harmless | Use |
|---|---|---|---|
| `direction` | 43 | 47 | extracting a refusal direction |
| `val` | 43 | 47 | tuning steering strength and thresholds |
| `test` | 113 | 156 | reported results only |

(Counts are per language.)

### Fields

| Field | Meaning |
|---|---|
| `id` | `<seed_id>_<language>` |
| `seed_id` | links the four language versions of one prompt |
| `language` | `en`, `hi`, `mr`, `hinglish` |
| `prompt` | the text |
| `harmful` | `true` for the HarmBench-derived half, `false` for the XSTest-derived half |
| `category` | source category (HarmBench risk area, or XSTest prompt type) |
| `source` | `HarmBench` or `XSTest` |
| `split` | `direction`, `val`, `test` |
| `hinglish_method` | for Hinglish rows: `llm` or `romanised` (see below) |

Harmful risk areas (unique prompts): illegal 57, cybercrime/intrusion 40,
misinformation/disinformation 34, chemical/biological 28, harmful (general) 21,
harassment/bullying 19.

## How it was built

1. **English seeds** come from HarmBench (harmful) and XSTest (harmless).
2. **Hindi and Marathi** are machine translations with NLLB-200-distilled-600M.
3. **Hinglish** is written by `CohereLabs/tiny-aya-base` from five few-shot examples.
   A base model is used on purpose: chat-tuned models answered the prompts instead of
   translating them. 443 of 449 Hinglish rows come from the model; 6 fell back to a
   rule-based romanisation of the Hindi (marked `romanised`).
4. **Cleaning** with CuratorKIT (exact and MinHash de-duplication over the English
   prompts). A prompt is kept only if all four language versions pass.

## Quality

- **Automated screening** of all 1,347 non-English rows (empty, untranslated, wrong
  script, extreme length, lost sentence ending, no shared terms): Hindi 100% pass,
  Marathi 100% pass, Hinglish 94.7% pass. 24 Hinglish rows were flagged for review.
- **Human review:** not yet completed. <!-- TODO: replace with the spot-check result -->
- Automated checks catch structural faults, not drift in meaning. A translation can
  pass every check and still soften a word (for example a violent verb rendered as a
  milder one), which matters most for the harmless-but-scary half.

## Intended use

Measuring whether a model's refusal behaviour survives translation and code-mixing,
and evaluating fixes. Not for training models to produce harmful content.

## Limitations

- **US-centric content.** The English seeds come from US benchmarks, so some prompts
  reference US institutions and are not typical of what Indian users write.
- **Machine-made translations.** Hinglish has no standard spelling, and the Hinglish
  here comes from a model in the same family as the models we tested, which could
  make the measured gap smaller than it is in real use.
- **Moderate size.** About 113 harmful prompts per language in the test split gives
  confidence intervals of roughly +/-9 points. Report intervals, not single numbers.
- **Languages.** Hindi, Marathi and Hinglish only.

## Ethics

The dataset contains **prompts only, never harmful completions**. The harmful prompts
are taken from a published benchmark and translated; none were written for this
project. It is intended for safety research.

## Licence and credits

Released under **CC BY-NC 4.0**. This is the most restrictive licence among the
components, chosen deliberately: the translations were produced by models released
under non-commercial licences (NLLB: CC BY-NC 4.0; Tiny Aya: CC BY-NC).

| Component | Source | Licence |
|---|---|---|
| Harmful English prompts | HarmBench (Mazeika et al., 2024, arXiv:2402.04249) | MIT |
| Harmless English prompts | XSTest (Rottger et al., 2023, arXiv:2308.01263) | CC BY 4.0 |
| Hindi and Marathi translation | NLLB-200-distilled-600M (Meta) | CC BY-NC 4.0 |
| Hinglish generation | CohereLabs/tiny-aya-base | CC BY-NC |
| Cleaning | CuratorKIT (Lexsi Labs) | see repository |

## Citation

If you use this dataset, please cite HarmBench and XSTest as well as this work.
