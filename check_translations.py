"""Automated quality check on the benchmark's translations.

Mechanical signals only, so the whole set can be screened without reading every
row. Writes two things:
  results/translation_quality.md   counts per language, for the write-up
  review/flagged_rows.xlsx         only the flagged rows, for a human pass

Run:  python check_translations.py [path/to/clean.csv]
"""
import re
import sys

import pandas as pd

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with",
    "how", "what", "why", "who", "is", "are", "do", "does", "did", "can", "i", "my",
    "me", "you", "your", "that", "this", "it", "as", "be", "from", "about", "into",
    "write", "give", "make", "create", "describe", "explain", "detailed", "step",
}


def content_words(text):
    """Longer, meaning-carrying words: the ones a translation should not silently drop."""
    words = re.findall(r"[a-z]{4,}", str(text).lower())
    return {w for w in words if w not in STOPWORDS}


def flag_row(english, translation, language):
    """Return a list of problem tags for one row (empty list = looks fine)."""
    en, tr = str(english).strip(), str(translation).strip()
    flags = []

    if not tr:
        return ["empty"]
    if tr.lower() == en.lower():
        flags.append("untranslated")
    if language == "hinglish" and DEVANAGARI.search(tr):
        flags.append("devanagari_in_hinglish")
    if language in ("hi", "mr") and not DEVANAGARI.search(tr):
        flags.append("no_devanagari")

    ratio = len(tr) / max(len(en), 1)
    if ratio < 0.5:
        flags.append("too_short")
    elif ratio > 2.2:
        flags.append("too_long")

    # A translation that keeps none of the English content words has probably
    # drifted; Hinglish normally borrows several of them outright.
    en_words = content_words(en)
    if language == "hinglish" and en_words:
        kept = len(en_words & content_words(tr)) / len(en_words)
        if kept == 0:
            flags.append("no_shared_terms")

    # Lost the source's terminal punctuation: a sign the text was cut off.
    # Most HarmBench prompts are imperatives with no final period, so a bare
    # ending only matters when the English actually had one.
    terminal = ".?!।"
    if en and en[-1] in terminal and tr[-1] not in terminal:
        flags.append("lost_sentence_end")

    return flags


def main(path=r"C:\Lexsi\kaggle\out1\data\clean.csv"):
    df = pd.read_csv(path)
    en = df[df["language"] == "en"].set_index("seed_id")["prompt"]
    df["english_original"] = df["seed_id"].map(en)
    rows = df[df["language"] != "en"].copy()

    rows["flags"] = [
        ";".join(flag_row(e, t, l))
        for e, t, l in zip(rows["english_original"], rows["prompt"], rows["language"])
    ]
    rows["ok"] = rows["flags"] == ""

    # Per-language pass rate, and how often each flag fires.
    summary = rows.groupby("language").agg(
        rows=("id", "size"), passed=("ok", "sum"), pass_rate=("ok", "mean"))
    tags = (rows.assign(tag=rows["flags"].str.split(";")).explode("tag")
            .query("tag != ''").groupby(["language", "tag"]).size().unstack(fill_value=0))

    with open(r"C:\Lexsi\results\translation_quality.md", "w", encoding="utf-8") as f:
        f.write("# Automated translation check\n\n")
        f.write(f"Screened {len(rows)} non-English rows "
                f"({rows['seed_id'].nunique()} prompts x 3 languages).\n\n")
        f.write("## Pass rate per language\n\n")
        f.write(summary.to_markdown(floatfmt=".1%") + "\n\n")
        f.write("## Flags raised (a row can raise more than one)\n\n")
        f.write(tags.to_markdown() + "\n")

    flagged = rows[~rows["ok"]][
        ["id", "language", "flags", "english_original", "prompt"]].copy()
    flagged["verdict"] = ""
    flagged["corrected_translation"] = ""
    flagged.to_excel(r"C:\Lexsi\review\flagged_rows.xlsx", index=False)

    print(summary.to_string(float_format="{:.1%}".format))
    print()
    print(tags.to_string())
    print(f"\n{len(flagged)} flagged rows -> review/flagged_rows.xlsx")


if __name__ == "__main__":
    main(*sys.argv[1:])
