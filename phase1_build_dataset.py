"""Phase 1: build the multilingual safety benchmark.

Steps
  1. Seed English prompts: HarmBench (harmful) + XSTest safe split (harmless but scary).
  2. Translate to Hindi and Marathi with NLLB (a translation model, so it never refuses).
  3. Make Hinglish by romanising the Hindi, plus a code-mixed variant.
  4. Push the raw set to Hugging Face, then clean it with CuratorKIT.

Only prompts are published, never harmful answers.
Run on a GPU; translation of ~600 prompts x 2 languages takes a few minutes on a T4.
"""
import random

import pandas as pd
from datasets import Dataset, load_dataset

from common import DATASET_CLEAN, DATASET_RAW, NLLB_CODES

N_HARMFUL = 300
N_HARMLESS = 300
SEED = 0


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


def to_hinglish(hindi, english):
    """Romanised Hindi; every other prompt also keeps a few English words (code-mixing)."""
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    roman = transliterate(hindi, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    if random.random() < 0.5:
        return roman
    # Code-mix: swap in the longest English words (usually the topic nouns).
    en_words = sorted(set(w.strip("?.,!") for w in english.split()), key=len, reverse=True)[:2]
    words = roman.split()
    for w in en_words:
        if words:
            words[random.randrange(len(words))] = w
    return " ".join(words)


def build():
    random.seed(SEED)
    df = seed_prompts()
    for lang, code in NLLB_CODES.items():
        df[lang] = translate(df["en"].tolist(), code)
    df["hinglish"] = [to_hinglish(h, e) for h, e in zip(df["hi"], df["en"])]

    # Long format: one row per (prompt, language).
    long = df.melt(id_vars=["seed_id", "harmful", "category", "source"],
                   value_vars=["en", "hi", "mr", "hinglish"],
                   var_name="language", value_name="prompt")
    long["id"] = long["seed_id"] + "_" + long["language"]

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


def curate():
    """Deduplicate and clean with CuratorKIT, then publish the final set."""
    from curatorkit import Curator, CuratorConfig

    result = Curator(CuratorConfig(
        dataset={"name": DATASET_RAW},
        dedup="minhash",
        clean=True,
        output_dir="data/curated",
    )).run()
    result.print_summary()
    # CuratorKIT writes rejected.jsonl and a dataset card next to the output;
    # inspect them, then push the cleaned set:
    #   load the curated output, then Dataset.push_to_hub(DATASET_CLEAN)
    print(f"Review data/curated, then push it to {DATASET_CLEAN}")


if __name__ == "__main__":
    build()
    curate()
