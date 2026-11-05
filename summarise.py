"""Turn judged answers into the result tables, with confidence intervals.

Runnable on its own against a saved judged CSV, so tables can be rebuilt
without spending GPU time:

    python summarise.py results/phase2_judged.csv results/phase2
"""
import sys

import pandas as pd

# 1.96 standard errors: the usual 95% interval.
Z = 1.96


def wilson(successes, n, z=Z):
    """95% confidence interval for a proportion (Wilson score interval).

    Better than the textbook formula at the extremes, which matters here:
    a refusal rate of 0% or 100% is common per language.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_with_ci(series):
    """Mean of a boolean series, plus its interval, as three columns."""
    n = len(series)
    k = int(series.sum())
    lo, hi = wilson(k, n)
    return pd.Series({"rate": k / n if n else float("nan"),
                      "ci_low": lo, "ci_high": hi, "n": n})


def summarise(logs, keys):
    """ASR and refusal (harmful prompts) + over-refusal (harmless), per `keys`.

    Either side may be empty (e.g. a harmful-only table), so empty sides are
    dropped rather than joined: a groupby-apply over no rows gives back the
    input columns, not the metric columns.
    """
    parts = []
    harmful, harmless = logs[logs["harmful"]], logs[~logs["harmful"]]
    if len(harmful):
        parts.append(harmful.groupby(keys).apply(lambda g: pd.concat([
            rate_with_ci(g["harmful_answer"]).add_prefix("asr_"),
            rate_with_ci(g["refused"]).add_prefix("refusal_"),
        ]), include_groups=False))
    if len(harmless):
        parts.append(harmless.groupby(keys).apply(
            lambda g: rate_with_ci(g["refused"]).add_prefix("overrefusal_"),
            include_groups=False))
    table = parts[0] if len(parts) == 1 else parts[0].join(parts[1], how="outer")
    return table.reset_index()


def write(table, path, title):
    table.to_csv(f"{path}.csv", index=False)
    with open(f"{path}.md", "w", encoding="utf-8") as f:
        f.write(f"## {title}\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(f"\n{title}\n{table.to_string(index=False)}")


def main(judged_csv, out_prefix):
    logs = pd.read_csv(judged_csv)

    # Headline: how each model behaves per language and attack.
    write(summarise(logs, ["model", "attack", "language"]),
          f"{out_prefix}_summary", "By model x attack x language")

    # Risk-area view: the gap may be far wider for some harm types. Harmful and
    # harmless prompts carry labels from different sources, so they get separate
    # tables (HarmBench risk areas vs XSTest over-refusal types).
    fire = logs[logs["model"].str.contains("fire") & (logs["attack"] == "plain")]
    write(summarise(fire[fire["harmful"]], ["category", "language"]),
          f"{out_prefix}_by_risk_area",
          "Tiny Aya Fire, plain prompts: harmful risk area x language")
    write(summarise(fire[~fire["harmful"]], ["category", "language"]),
          f"{out_prefix}_by_safe_type",
          "Tiny Aya Fire, plain prompts: over-refusal by prompt type x language")


if __name__ == "__main__":
    main(*sys.argv[1:])
