"""Publish the project's artifacts to the Hugging Face Hub.

    python publish_hf.py datasets     benchmark public + card; public redacted eval logs
    python publish_hf.py steering     steering direction + gate as a model repo

Raw generation logs stay private: they contain answers the judge scored as
harmful. The public eval-logs dataset keeps every prompt, verdict and metric
but replaces the text of answers judged harmful, so all reported numbers can
be recomputed without redistributing harmful completions.
"""
import sys

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

from common import DATASET_CLEAN, DATASET_RAW, HF_USER
from summarise import is_broken

EVAL_LOGS = f"{HF_USER}/hinglish-safety-eval-logs"
STEERING_REPO = f"{HF_USER}/tiny-aya-fire-hinglish-refusal-steering"
REDACTED = "[redacted: judged harmful]"

RUNS = {
    # config name: (judged csv, rows to keep)
    "baseline_test": ("results/phase2_judged.csv", None),
    "steering_val": ("results/phase4_val_judged.csv", None),
    "steering_test": ("results/phase4_judged.csv", None),
}
KEEP = ["id", "seed_id", "language", "split", "harmful", "category", "source", "model",
        "attack", "prompt", "prompt_en", "answer", "answer_en", "refused",
        "harmful_answer", "broken", "judge_parsed"]


def redact(df):
    df = df.copy()
    df["broken"] = df["answer"].map(is_broken)   # computed before redaction
    for col in ("answer", "answer_en"):
        df.loc[df["harmful_answer"], col] = REDACTED
    return df[[c for c in KEEP if c in df.columns]]


EVAL_CARD = f"""---
license: cc-by-nc-4.0
language: [en, hi, mr]
tags: [safety, red-teaming, multilingual, code-mixed, hinglish, evaluation]
pretty_name: Hinglish Safety Eval Logs
---

# Hinglish Safety Eval Logs

Judged model outputs behind every number in *Safety Doesn't Translate*
(Lexsi AI Safety and Alignment Hackathon, Track 2a). Prompts come from
[{DATASET_CLEAN}](https://huggingface.co/datasets/{DATASET_CLEAN}).

| Config | What it is |
|---|---|
| `baseline_test` | Tiny Aya Fire and Global, plain and fiction-framed prompts, test split |
| `steering_val` | candidate steering settings on the val split (used to choose one) |
| `steering_test` | chosen conditional steering and its no-gate twin, test split |

Each row has the WildGuard verdicts (`refused`, `harmful_answer`), a
`broken` flag (near-empty or looping answer) and the English text the judge
read (`answer_en`; Hindi and Marathi answers were machine-translated).

**Redaction.** Answers the judge scored as harmful are replaced with
`{REDACTED}`. Verdicts are kept, so every rate in the write-up can be
recomputed from this dataset. The unredacted logs are not published.

Code: https://github.com/kdt523/safety-doesnt-translate
"""


def datasets():
    api = HfApi()
    api.upload_file(path_or_fileobj="cards/DATASET_CARD.md", path_in_repo="README.md",
                    repo_id=DATASET_CLEAN, repo_type="dataset",
                    commit_message="Dataset card")
    api.update_repo_settings(DATASET_CLEAN, repo_type="dataset", private=False)
    print(f"public: {DATASET_CLEAN}")

    for config, (path, _) in RUNS.items():
        df = redact(pd.read_csv(path))
        n = int((df["answer"] == REDACTED).sum())
        Dataset.from_pandas(df, preserve_index=False).push_to_hub(
            EVAL_LOGS, config, private=False)
        print(f"  {config}: {len(df)} rows, {n} answers redacted")
    api.upload_file(path_or_fileobj=EVAL_CARD.encode(), path_in_repo="README.md",
                    repo_id=EVAL_LOGS, repo_type="dataset", commit_message="Card")
    print(f"public: {EVAL_LOGS}")
    print(f"kept private: {DATASET_RAW}, {HF_USER}/hinglish-safety-logs")


def steering():
    api = HfApi()
    api.create_repo(STEERING_REPO, exist_ok=True)
    for f in ("phase3_directions.pt", "phase4_condition.pt", "phase4_ctx.json",
              "phase4_choice.json", "phase3_pca.png"):
        api.upload_file(path_or_fileobj=f"results/{f}", path_in_repo=f,
                        repo_id=STEERING_REPO)
    api.upload_file(path_or_fileobj="cards/STEERING_CARD.md", path_in_repo="README.md",
                    repo_id=STEERING_REPO)
    print(f"public: {STEERING_REPO}")


if __name__ == "__main__":
    {"datasets": datasets, "steering": steering}[sys.argv[1]]()
