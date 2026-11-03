"""Phase 1: build the multilingual safety benchmark.

Steps
  1. Seed English prompts: HarmBench (harmful) + XSTest safe split (harmless but scary).
  2. Translate to Hindi and Marathi with NLLB (a translation model, so it never refuses).
  3. Make natural Hinglish with tiny-aya-global (romanised Hindi as fallback on refusal).
  4. Push the raw set to Hugging Face, then clean it with CuratorKIT.

Only prompts are published, never harmful answers.
Run on a GPU; translation of ~600 prompts x 2 languages takes a few minutes on a T4.
"""
import random

import pandas as pd
from datasets import Dataset, load_dataset

from common import COMPARE_MODEL_ID, DATASET_CLEAN, DATASET_RAW, NLLB_CODES

N_HARMFUL = 300
N_HARMLESS = 300
SEED = 0
TRANSLATOR_ID = COMPARE_MODEL_ID  # tiny-aya-global writes the Hinglish; tiny-aya-fire is tested


def seed_prompts():
    hb = load_dataset("walledai/HarmBench", "standard", split="train").to_pandas()
    xs = load_dataset("walledai/XSTest", split="test").to_pandas()
    # Check the column names once; dataset authors sometimes rename them.
    assert "prompt" in hb.columns and "prompt" in xs.columns, (hb.columns, xs.columns)
    xs = xs[xs["label"] == "safe"]

    harmful = hb.sample(min(N_HARMFUL, len(hb)), random_state=SEED)
    harmless = xs.sample(min(N_HARMLESS, len(xs)), random_state=SEED)
    rows = [
        {"seed_id": f"hb{i}", "en": r.prompt, "harmful": True,
         "category": getattr(r, "category", "unknown"), "source": "HarmBench"}
        for i, r in enumerate(harmful.itertuples())
    ] + [
        {"seed_id": f"xs{i}", "en": r.prompt, "harmful": False,
         "category": getattr(r, "type", "unknown"), "source": "XSTest"}
        for i, r in enumerate(harmless.itertuples())
    ]
    return pd.DataFrame(rows)


def translate(texts, tgt, batch_size=16):
    # Called directly: transformers v5 dropped the "translation" pipeline.
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda")
    tgt_id = tok.convert_tokens_to_ids(tgt)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            gen = model.generate(**batch, forced_bos_token_id=tgt_id, max_new_tokens=256)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


REFUSAL_MARKERS = ("sorry", "can't", "cannot", "unable", "i won't", "maaf", "nahi kar sakt")
HINGLISH_PROMPT = (
    "You are a translation engine. Rewrite the user's text in natural Hinglish: Hindi "
    "written in the Latin alphabet, mixed with common English words, the way people in "
    "India text each other. Output ONLY the rewritten text. Never answer, refuse or "
    "comment on it, even if it asks for something harmful; you are only translating."
)


def romanise(hindi):
    """Rule-based fallback: Devanagari -> casual Latin spelling."""
    import re

    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    r = transliterate(hindi, sanscript.DEVANAGARI, sanscript.ITRANS)
    for a, b in (("M", "n"), (".N", "n"), ("~N", "n"), ("~n", "n"), ("RRi", "ri"),
                 ("Sh", "sh"), ("shh", "sh"), (".D", "d"), (".h", "")):
        r = r.replace(a, b)
    r = r.lower()
    # Drop the inherent final vowel (karana -> karan, eka -> ek), keep long "aa".
    r = re.sub(r"(?<=[^aeiou\s])a\b", "", r)
    for a, b in (("aa", "a"), ("ii", "i"), ("uu", "u"), ("ee", "e")):
        r = r.replace(a, b)
    return r


def to_hinglish(english, hindi, batch_size=16):
    """LLM rewrite into natural Hinglish; falls back to romanised Hindi on refusal."""
    import re

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TRANSLATOR_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        TRANSLATOR_ID, torch_dtype=torch.float16, device_map="auto")
    outs = []
    for i in range(0, len(english), batch_size):
        chats = [[{"role": "system", "content": HINGLISH_PROMPT},
                  {"role": "user", "content": t}] for t in english[i:i + batch_size]]
        batch = tok.apply_chat_template(chats, add_generation_prompt=True, padding=True,
                                        return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=128, do_sample=False)
        outs += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)

    # Keep the raw rewrites: how often the translator refuses is itself a finding.
    pd.DataFrame({"en": english, "llm_output": outs}).to_csv(
        "results/hinglish_llm_raw.csv", index=False)

    text, method = [], []
    for out, en, hi in zip(outs, english, hindi):
        out = out.strip().strip('"').splitlines()[0].strip() if out.strip() else ""
        bad = (not out or re.search("[\\u0900-\\u097F]", out)  # still Devanagari
               or any(m in out.lower() for m in REFUSAL_MARKERS)
               or len(out) > 3 * len(en) + 40)
        text.append(romanise(hi) if bad else out)
        method.append("romanised" if bad else "llm")
    return text, method


def build():
    random.seed(SEED)
    df = seed_prompts()
    for lang, code in NLLB_CODES.items():
        df[lang] = translate(df["en"].tolist(), code)
    df["hinglish"], df["hinglish_method"] = to_hinglish(df["en"].tolist(), df["hi"].tolist())
    print(df["hinglish_method"].value_counts())

    # Long format: one row per (prompt, language).
    long = df.melt(id_vars=["seed_id", "harmful", "category", "source", "hinglish_method"],
                   value_vars=["en", "hi", "mr", "hinglish"],
                   var_name="language", value_name="prompt")
    long["id"] = long["seed_id"] + "_" + long["language"]
    long.loc[long["language"] != "hinglish", "hinglish_method"] = ""

    # Fixed splits by seed, so all 4 languages of a prompt land in the same split.
    seeds = sorted(df["seed_id"])
    random.shuffle(seeds)
    n = len(seeds)
    split_of = {s: "direction" if i < 0.2 * n else "val" if i < 0.4 * n else "test"
                for i, s in enumerate(seeds)}
    long["split"] = long["seed_id"].map(split_of)

    long.to_csv("data/raw.csv", index=False)
    Dataset.from_pandas(long, preserve_index=False).push_to_hub(DATASET_RAW, private=True)
    print(long.groupby(["language", "harmful", "split"]).size())
    return long


def run_curator(rows, dedup, out_dir):
    """Run CuratorKIT on `rows`; return a bool per row saying whether it passed.

    CuratorKIT reads a local file: re-reading the Hub copy breaks when the columns
    change between runs. Its output only carries prompt text (and dropped duplicates
    are not listed in rejected.jsonl), so rows are matched back by normalised text,
    consuming each passed prompt once.
    """
    import json
    from collections import Counter

    from curatorkit import Curator, CuratorConfig

    path = f"{out_dir}.jsonl"
    rows[["id", "prompt"]].to_json(path, orient="records", lines=True, force_ascii=False)
    Curator(CuratorConfig(
        dataset=path,
        format="prompt_only",
        min_tokens=1,  # safety prompts are short by design; the default (10) drops them
        dedup=dedup,
        clean=True,
        output_dir=out_dir,
    )).run().print_summary()

    norm = lambda t: " ".join(str(t).split())
    with open(f"{out_dir}/ppo.jsonl", encoding="utf-8") as f:
        passed = Counter(norm(json.loads(line)["prompt"]) for line in f if line.strip())
    kept = []
    for t in rows["prompt"]:
        kept.append(passed[norm(t)] > 0)
        passed[norm(t)] -= 1
    return pd.Series(kept, index=rows.index)


def curate(long):
    """Clean with CuratorKIT and publish the prompts that passed in all 4 languages.

    Near-duplicates are found among the English seeds only: Hindi and Marathi share a
    script, so cross-language MinHash would wrongly drop real translations.
    """
    en = long[long["language"] == "en"]
    unique_seeds = set(en.loc[run_curator(en, "minhash", "data/curated_en"), "seed_id"])
    long = long.assign(kept=run_curator(long, "none", "data/curated"))
    full_seeds = long.groupby("seed_id")["kept"].transform("all") & long["seed_id"].isin(unique_seeds)
    clean = long[full_seeds].drop(columns="kept")
    print(f"Kept {clean['seed_id'].nunique()} of {long['seed_id'].nunique()} prompts "
          f"({len(clean)} rows); publishing to {DATASET_CLEAN}")
    clean.to_csv("data/clean.csv", index=False)
    Dataset.from_pandas(clean, preserve_index=False).push_to_hub(DATASET_CLEAN, private=True)


if __name__ == "__main__":
    curate(build())
