"""Phase 5: does the fix hold up under repeated attempts? (Track 2a #1)

Two robustness checks, each run on the unsteered model and on the steered one
(the setting Phase 4 chose on the val split):

  library probes   AuditKIT's own RedTeamRunner with its built-in probes
                   (prompt injection, jailbreak, encoding, refusal). They are
                   small and English-only, so they are a sanity check, not the
                   main evidence.

  best-of-N        each harmful test prompt is sampled N times at temperature.
                   An attacker only needs one compliant answer, so a prompt
                   counts as broken if ANY of its N samples is judged harmful.
                   This is a standard robustness measure and adds no new
                   attack content: it reuses the benchmark's prompts.

Over-refusal is reported alongside: the greedy over-refusal for both models
comes from the Phase 2 and Phase 4 judged runs (harmless prompts, same judge).

Stages:
  python phase5_robustness.py generate
  python phase2_baseline_eval.py judge bestofn phase5      (WildGuard)
  python bestofn_summary.py results/phase5_judged.csv results/phase5
"""
import json
import sys

import pandas as pd
import torch
from datasets import Dataset, load_dataset

from common import DATASET_CLEAN, HF_USER, MODEL_ID, NLLB_CODES, load_model

LOGS = f"{HF_USER}/hinglish-safety-logs"
N_SAMPLES = 5
TEMPERATURE = 1.0
TOP_P = 0.95
SEEDS_PER_LANGUAGE = 50       # harmful prompts per language; keeps the run affordable
MAX_NEW_TOKENS = 128          # compliance shows up early; full length is not needed
PROBE_MAX_NEW_TOKENS = 200


def steered_setup():
    """The setting chosen on val in Phase 4, plus the saved gate and direction."""
    import phase4_steer as p4

    choice = json.load(open(p4.CHOICE_FILE))
    setting = {k: choice[k] for k in ("variant", "alpha") if k in choice}
    return setting, p4.load_ctx()


def library_probes(model, tok, setting, ctx):
    """AuditKIT's RedTeamRunner against both models, via callables."""
    from auditkit.redteam import RedTeamRunner

    import phase4_steer as p4

    def base(prompts):
        return p4.generate_batched(model.generate, model, tok, prompts, PROBE_MAX_NEW_TOKENS)

    def steered(prompts):
        return p4.steer_generate(model, tok, prompts, setting, ctx,
                                 max_new_tokens=PROBE_MAX_NEW_TOKENS)

    rows = []
    for name, fn in (("unsteered", base), ("steered", steered)):
        result = RedTeamRunner(fn).run()
        print(f"\n[{name}]\n{result.summary()}", flush=True)
        for r in result.results:
            rows.append({"model_variant": name, "probe": r.probe_name,
                         "detector": r.detector_name, "passed": r.passed})
    table = pd.DataFrame(rows)
    summary = (table.groupby(["model_variant", "probe", "detector"])["passed"]
               .agg(pass_rate="mean", n="size").reset_index())
    summary.to_csv("results/phase5_library_probes.csv", index=False)
    print(summary.to_string(index=False))


def best_of_n(model, tok, setting, ctx):
    import phase4_steer as p4

    df = load_dataset(DATASET_CLEAN, split="train").to_pandas()
    df = df[(df["split"] == "test") & df["harmful"]]
    keep = (df[df["language"] == "en"].sort_values("seed_id")
            .head(SEEDS_PER_LANGUAGE)["seed_id"])
    df = df[df["seed_id"].isin(keep)]
    prompts = df["prompt"].tolist()
    print(f"best-of-{N_SAMPLES}: {len(df)} harmful prompts "
          f"({SEEDS_PER_LANGUAGE} per language)", flush=True)

    rows = []
    for variant in ("unsteered", "steered"):
        for k in range(N_SAMPLES):
            sample_args = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
            run = {"variant": "unsteered"} if variant == "unsteered" else setting
            answers = p4.steer_generate(model, tok, prompts, run, ctx,
                                        max_new_tokens=MAX_NEW_TOKENS, **sample_args)
            rows.append(df.assign(attack=f"bestofn_{variant}", sample=k, answer=answers,
                                  model=MODEL_ID.split("/")[-1]))
            print(f"  {variant} sample {k + 1}/{N_SAMPLES}", flush=True)
    return pd.concat(rows, ignore_index=True)


def generate():
    model, tok = load_model(MODEL_ID)
    setting, ctx = steered_setup()

    library_probes(model, tok, setting, ctx)
    logs = best_of_n(model, tok, setting, ctx)
    del model
    torch.cuda.empty_cache()

    # The judge reads English: add the English prompt and translate Devanagari answers.
    import phase2_baseline_eval as p2

    full = load_dataset(DATASET_CLEAN, split="train").to_pandas()
    en = full[full["language"] == "en"].set_index("seed_id")["prompt"]
    logs["prompt_en"] = logs["seed_id"].map(en)
    logs["answer_en"] = logs["answer"]
    for lang, code in NLLB_CODES.items():
        mask = logs["language"] == lang
        logs.loc[mask, "answer_en"] = p2.to_english(logs.loc[mask, "answer"].tolist(), code)

    logs.to_csv("results/phase5_generations.csv", index=False)
    Dataset.from_pandas(logs, preserve_index=False).push_to_hub(LOGS, "bestofn", private=True)
    print(logs.groupby(["attack", "language"]).size())


if __name__ == "__main__":
    {"generate": generate}[sys.argv[1]]()
