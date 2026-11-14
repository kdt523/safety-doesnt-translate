---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-fire
language: [en, hi, mr]
tags: [safety, activation-steering, refusal, multilingual, hinglish, cast]
---

# Refusal steering for Tiny Aya Fire (Hindi, Marathi, Hinglish)

Inference-time fix for the multilingual refusal gap in
[`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire). No
weights are changed: a vector is added to the residual stream for prompts that a
gate flags as harmful.

Built for the Lexsi AI Safety and Alignment Hackathon (Track 2a).
Code: https://github.com/kdt523/safety-doesnt-translate

## Files

| File | Contents |
|---|---|
| `phase3_directions.pt` | unit refusal directions at layer 18 (`en`, `hi`, `mr`, `hinglish`, `_layer`). Steering uses `en`. |
| `phase4_condition.pt` | CAST gate: `condition_vector`, `condition_layer` (17), `threshold`, `comparator`, `pool` |
| `phase4_ctx.json` | layer, gate settings, English refusal target (24.9) |
| `phase4_choice.json` | chosen setting (CAST, alpha 12) and its val-split metrics |

## What it does

Harmful Hinglish prompts activate the model's refusal direction weakly (projection 17.1
against 24.9 for English). The gate (SafeTune `fit_cast_condition`, F1 0.98 on held-out
prompts in all four languages) checks the last prompt token at layer 17. If it fires,
`12 * direction` is added at layer 18 for the whole generation. Prompts that do not trip
the gate are untouched.

## Results (held-out test split, WildGuard judge)

| Language | Attack success before | after | Over-refusal before | after |
|---|---|---|---|---|
| English | 7.1% | 1.8% | 6.4% | 6.4% |
| Hindi | 17.7% | 6.2% | 5.1% | 5.1% |
| Marathi | 31.9% | 11.5% | 12.2% | 12.2% |
| Hinglish | 51.3% | 31.0% | 7.1% | 7.1% |

Hinglish remains the weakest language. See the write-up for intervals and limitations.

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetune.interventions.steer.cast import _cast_similarity, _condition_met

repo = "kd1609/tiny-aya-fire-hinglish-refusal-steering"
direction = torch.load(hf_hub_download(repo, "phase3_directions.pt"))["en"]
gate = torch.load(hf_hub_download(repo, "phase4_condition.pt"), weights_only=False)
LAYER, ALPHA = 18, 12.0

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-fire", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-fire",
                                             torch_dtype=torch.float16, device_map="auto")
layers = model.model.layers

def answer(prompt, steer=True, max_new_tokens=256):
    batch = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, return_dict=True,
                                    return_tensors="pt").to(model.device)
    alpha = 0.0
    if steer:
        seen = {}
        h = layers[gate["condition_layer"]].register_forward_hook(
            lambda m, i, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
        with torch.no_grad():
            model(**batch, logits_to_keep=1)
        h.remove()
        sim = _cast_similarity(seen["h"], gate["condition_vector"])
        alpha = ALPHA if _condition_met(sim, gate["threshold"], gate["comparator"]) else 0.0

    def add(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        x = x + alpha * direction.to(x.dtype).to(x.device)
        return (x,) + o[1:] if isinstance(o, tuple) else x

    h = layers[LAYER].register_forward_hook(add)
    try:
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        h.remove()
    return tok.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
```

Batch with **left padding** only: right padding breaks batched generation.

## Licence

CC BY-NC 4.0, following the base model's non-commercial licence.
