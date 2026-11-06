"""Phase 3: locate the refusal direction and ask why Hinglish slips past it.

Phase 2 showed Tiny Aya Fire refuses 91% of harmful English prompts but only
35% of the same prompts in Hinglish. Two explanations are possible:

  (a) weak activation - Hinglish prompts barely register on the refusal
      direction the model already has, so refusal never fires;
  (b) rotation - Hinglish has its own, different refusal direction, and the
      English one does not describe it.

Which one holds decides the Phase 4 fix: (a) wants the existing direction
pushed harder, (b) wants a language-aware direction.

Method (Arditi et al. 2406.11717, via SafeTune):
  direction = mean(harmful activations) - mean(harmless activations)
at the last prompt token, taken on the held-out `direction` split only.

Outputs
  results/phase3_projection.csv   how strongly each language sits on the English direction
  results/phase3_cosine.csv       cosine similarity between per-language directions
  results/phase3_directions.pt    the vectors themselves (Phase 4 + Hub upload)
  results/phase3_pca.png          activations in 2-D, coloured by language and harmfulness
"""
import json

import pandas as pd
import torch
from datasets import load_dataset

from common import DATASET_CLEAN, LANGS, MODEL_ID, load_model

SPLIT = "direction"


def prompts_by_language(df, language):
    sub = df[df["language"] == language]
    return (sub[sub["harmful"]]["prompt"].tolist(),
            sub[~sub["harmful"]]["prompt"].tolist())


def activations(model, tok, prompts, layers):
    """Pooled residual-stream activations per layer: {layer: (N, d)}.

    Uses SafeTune's own extractor so the pooling matches what
    extract_refusal_direction did when it built the direction. The collector is
    private, but it is the only way to get per-prompt activations rather than a
    single averaged vector.
    """
    from safetune.core.runtime.inference.vector_extraction import (
        SteeringVectorExtractor, VectorExtractionConfig)

    cfg = VectorExtractionConfig(target_layers=layers, batch_size=8,
                                 pool_method="last_token", normalize=False)
    extractor = SteeringVectorExtractor(model, tok, config=cfg)
    return extractor._collect_activations(prompts)


def main():
    from safetune.interventions.steer import (RefusalDirectionConfig,
                                              extract_refusal_direction)

    df = load_dataset(DATASET_CLEAN, split="train").to_pandas()
    df = df[df["split"] == SPLIT]
    model, tok = load_model(MODEL_ID)

    # 1. The model's own refusal direction, found on English contrast pairs.
    en_harmful, en_harmless = prompts_by_language(df, "en")
    print(f"Extracting direction from {len(en_harmful)} harmful / "
          f"{len(en_harmless)} harmless English prompts", flush=True)
    direction, layer, per_layer = extract_refusal_direction(
        model, tok, en_harmful, en_harmless,
        RefusalDirectionConfig(pool_method="last_token"))
    print(f"Refusal direction selected at layer {layer}", flush=True)

    direction = direction.float().cpu()
    store = {"en": direction, "_layer": layer}

    # 2. Where each language's prompts sit along that direction. A harmful
    #    prompt the model recognises should score clearly above a harmless one.
    rows = []
    acts = {}
    for lang in LANGS:
        harmful, harmless = prompts_by_language(df, lang)
        a_harmful = activations(model, tok, harmful, [layer])[layer].float().cpu()
        a_harmless = activations(model, tok, harmless, [layer])[layer].float().cpu()
        acts[lang] = (a_harmful, a_harmless)

        for name, acts_ in (("harmful", a_harmful), ("harmless", a_harmless)):
            proj = acts_ @ direction
            rows.append({"language": lang, "prompts": name, "n": len(proj),
                         "mean_projection": proj.mean().item(),
                         "std_projection": proj.std().item()})

        # 3. That language's own refusal direction, for comparison with English.
        own = (a_harmful.mean(0) - a_harmless.mean(0))
        store[lang] = own / own.norm()

    proj_table = pd.DataFrame(rows)
    # Separation = how far harmful sits above harmless, in standard deviations.
    sep = (proj_table.pivot(index="language", columns="prompts",
                            values="mean_projection")
           .assign(separation=lambda t: t["harmful"] - t["harmless"]))
    sep["separation_sd"] = sep["separation"] / proj_table.groupby("language")[
        "std_projection"].mean().values
    proj_table.to_csv("results/phase3_projection.csv", index=False)
    sep.to_csv("results/phase3_separation.csv")
    print("\nProjection onto the English refusal direction:")
    print(sep.to_string(float_format="{:.3f}".format))

    # 4. Do the languages even point the same way?
    cos = pd.DataFrame(
        [[torch.dot(store[a], store[b]).item() for b in LANGS] for a in LANGS],
        index=LANGS, columns=LANGS)
    cos.to_csv("results/phase3_cosine.csv")
    print("\nCosine similarity between per-language refusal directions:")
    print(cos.to_string(float_format="{:.3f}".format))

    torch.save(store, "results/phase3_directions.pt")
    with open("results/phase3_meta.json", "w") as f:
        json.dump({"model": MODEL_ID, "layer": layer, "split": SPLIT,
                   "n_harmful_en": len(en_harmful), "n_harmless_en": len(en_harmless)}, f, indent=2)

    plot(acts, direction)


def plot(acts, direction):
    """Activations projected to 2-D: refusal direction on x, main spread on y."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    every = torch.cat([a for pair in acts.values() for a in pair])
    centred = every - every.mean(0)
    # Second axis: the largest direction of variation that is not refusal.
    residual = centred - torch.outer(centred @ direction, direction)
    second = torch.linalg.svd(residual, full_matrices=False).Vh[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = dict(zip(LANGS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
    for lang, (a_harmful, a_harmless) in acts.items():
        for acts_, marker, label in ((a_harmful, "^", "harmful"), (a_harmless, "o", "harmless")):
            centred_ = acts_ - every.mean(0)
            ax.scatter(centred_ @ direction, centred_ @ second, s=18, alpha=0.6,
                       marker=marker, color=colours[lang],
                       label=f"{lang} {label}")
    ax.set_xlabel("projection on English refusal direction")
    ax.set_ylabel("second principal axis")
    ax.set_title("Where each language's prompts sit relative to refusal")
    ax.legend(fontsize=7, ncol=2)
    ax.axvline(0, color="grey", lw=0.5)
    fig.tight_layout()
    fig.savefig("results/phase3_pca.png", dpi=150)
    print("\nWrote results/phase3_pca.png")


if __name__ == "__main__":
    main()
