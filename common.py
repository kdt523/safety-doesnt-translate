"""Shared settings for every phase."""
import os

# Main model: South-Asia-focused Tiny Aya. Global is the comparison model.
MODEL_ID = os.environ.get("MODEL_ID", "CohereLabs/tiny-aya-fire")
COMPARE_MODEL_ID = "CohereLabs/tiny-aya-global"

# Your Hugging Face username; every artifact is pushed under it.
HF_USER = os.environ.get("HF_USER", "kd1609")
DATASET_RAW = f"{HF_USER}/hinglish-safety-bench-raw"
DATASET_CLEAN = f"{HF_USER}/hinglish-safety-bench"

LANGS = ["en", "hi", "mr", "hinglish"]
NLLB_CODES = {"hi": "hin_Deva", "mr": "mar_Deva"}


def load_model(model_id=MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Left padding is required for batched generation: with right padding, every
    # shorter prompt in a batch has pad tokens between its prompt and its answer,
    # and the model writes broken text. (This corrupted the first Phase 4 run.)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def chat(model, tok, prompt, max_new_tokens=200):
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
