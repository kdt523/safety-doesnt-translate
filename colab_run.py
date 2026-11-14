"""One-shot runner for Google Colab (free T4), used when Kaggle GPU quota ran out.

Runs Phase 6 (multi-turn audits) then Phase 5 (best-of-N + probes). After every
step, results/ is uploaded to a private Hub dataset, so a failure in a later step
(e.g. the 7B judge running out of memory on one T4) never loses earlier results.

In Colab: Runtime -> T4 GPU, add a secret HF_TOKEN, then run the cell in README.
"""
import os
import subprocess
import sys

RESULTS_REPO = f"{os.environ.get('HF_USER', 'kd1609')}/lexsi-run-results"

PHASE6 = ["python phase6_multiturn.py generate",
          "python phase6_multiturn.py judge"]
PHASE5 = ["python phase5_robustness.py generate",
          "python phase2_baseline_eval.py judge bestofn phase5",
          "python bestofn_summary.py results/phase5_judged.csv results/phase5"]
# `python colab_run.py 6` or `5` runs one phase (two jobs in parallel); no argument runs both.
STEPS = {"6": PHASE6, "5": PHASE5}.get(sys.argv[1] if len(sys.argv) > 1 else "", PHASE6 + PHASE5)


def upload(note):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path="results", repo_id=RESULTS_REPO, repo_type="dataset",
                      allow_patterns=["phase5_*", "phase6_*"], commit_message=note)
    print(f"uploaded results after: {note}", flush=True)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])
    for step in STEPS:
        print(f"\n$ {step}", flush=True)
        code = subprocess.run(step, shell=True).returncode
        print(f"exit {code}", flush=True)
        try:
            upload(step)
        except Exception as e:
            print(f"upload failed: {e}", flush=True)
        if code and "generate" in step:
            # Judging needs the generations; skip to the next phase.
            print("generation failed; skipping its judge step", flush=True)
            idx = STEPS.index(step)
            while idx + 1 < len(STEPS) and "generate" not in STEPS[idx + 1]:
                STEPS.pop(idx + 1)


if __name__ == "__main__":
    sys.exit(main())
