"""Build the static Hugging Face Space (index.html) from the local results.

Hugging Face requires a paid plan to host Gradio Spaces, so the public demo is a
static page: results, mechanism, and before/after answers (harmful answers
redacted). The live gate inspector (app.py) needs a paid or GPU Space.

    python space/build_static.py      -> space/static/index.html
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from summarise import is_broken, wilson  # noqa: E402

R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "space", "static")
REDACTED = "[redacted: judged harmful]"
LANG = {"en": "English", "hi": "Hindi", "mr": "Marathi", "hinglish": "Hinglish"}


def rates(df, col):
    k, n = int(df[col].sum()), len(df)
    lo, hi = wilson(k, n)
    return {"rate": round(100 * k / n, 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}


def main():
    base = pd.read_csv(f"{R}/phase2_judged.csv")
    base = base[(base["model"] == "tiny-aya-fire") & (base["attack"] == "plain")]
    steer = pd.read_csv(f"{R}/phase4_judged.csv")
    runs = {"before": base, "cast": steer[steer["attack"].str.startswith("cast")],
            "always": steer[steer["attack"].str.startswith("always")]}

    table = []
    for lang, name in LANG.items():
        row = {"language": name}
        for run, df in runs.items():
            d = df[df["language"] == lang]
            row[run] = {"asr": rates(d[d["harmful"]], "harmful_answer"),
                        "over": rates(d[~d["harmful"]], "refused")}
        table.append(row)

    proj = pd.read_csv(f"{R}/phase3_projection.csv")
    mech = [{"language": LANG[r.language], "prompts": r.prompts,
             "mean": round(r.mean_projection, 1)} for r in proj.itertuples()]

    b = base.set_index("id")
    c = runs["cast"].set_index("id")
    pairs = []
    for lang in LANG:
        for harmful in (True, False):
            ids = [i for i in c.index if c.loc[i, "language"] == lang
                   and bool(c.loc[i, "harmful"]) == harmful and i in b.index]
            for i in sorted(ids)[:25]:
                def text(row):
                    if row["harmful_answer"]:
                        return REDACTED
                    t = row["answer"] if isinstance(row["answer"], str) else ""
                    return t[:600]
                pairs.append({"lang": lang, "harmful": harmful,
                              "prompt": b.loc[i, "prompt"][:300],
                              "before": text(b.loc[i]), "after": text(c.loc[i]),
                              "before_v": verdict(b.loc[i]), "after_v": verdict(c.loc[i])})

    bon = pd.read_csv(f"{R}/phase5_bestofn.csv")
    fic = pd.read_csv(f"{R}/phase7_fiction_summary.csv")
    esc = pd.read_csv(f"{R}/phase6_escalation_summary.csv")
    inj = pd.read_csv(f"{R}/phase6_injection_summary.csv")
    robust = []
    for lang, name in LANG.items():
        def pick(df, col, **eq):
            row = df
            for k, v in eq.items():
                row = row[row[k] == v]
            return round(100 * float(row[col].iloc[0]), 1)
        robust.append({
            "language": name,
            "fiction": [pick(fic, "asr_rate", variant="unsteered", language=lang),
                        pick(fic, "asr_rate", variant="steered", language=lang)],
            "multiturn": [pick(esc, "multi_turn_asr", variant="unsteered", language=lang),
                          pick(esc, "multi_turn_asr", variant="steered", language=lang)],
            "bestofn": [pick(bon, "best_of_n_asr", variant="unsteered", language=lang),
                        pick(bon, "best_of_n_asr", variant="steered", language=lang)],
            "injection": [pick(inj, "leaked", variant="unsteered", kind="injected",
                               injection_language=lang),
                          pick(inj, "leaked", variant="steered", kind="injected",
                               injection_language=lang)],
        })
    data = {"table": table, "mech": mech, "pairs": pairs, "robust": robust}
    os.makedirs(OUT, exist_ok=True)
    html = open(os.path.join(ROOT, "space", "template.html"), encoding="utf-8").read()
    html = html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
    print(f"wrote {OUT}/index.html: {len(pairs)} answer pairs")


def verdict(row):
    if row["harmful_answer"]:
        return "harmful"
    if is_broken(row["answer"]):
        return "broken"
    return "refused" if row["refused"] else "answered"


if __name__ == "__main__":
    main()
