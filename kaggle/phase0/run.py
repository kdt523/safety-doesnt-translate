"""Kaggle runner: pull the latest code from GitHub and run one phase.

Pushed from the laptop with:
  python -m kaggle kernels push -p kaggle/phase0 --accelerator NvidiaTeslaT4
Outputs land in /kaggle/working/results (downloaded with `kaggle kernels output`).
"""
import os
import shutil
import subprocess
import sys

PHASE = "phase0_smoke_test.py"
REPO = "https://github.com/kdt523/safety-doesnt-translate.git"
WORK = "/kaggle/working"
PROJECT = "/tmp/project"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


print("python", sys.version)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

shutil.rmtree(PROJECT, ignore_errors=True)
sh(f"git clone -q --depth 1 {REPO} {PROJECT}")
os.chdir(PROJECT)

# Quiet install; a failure of one Lexsi package is reported but does not stop phase 0.
if sh("pip install -q -r requirements.txt > /tmp/pip.log 2>&1"):
    print("Full install failed; last lines of pip log:")
    sh("tail -n 15 /tmp/pip.log")
    sh("pip install -q -U transformers accelerate datasets huggingface_hub")

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login

    login(UserSecretsClient().get_secret("HF_TOKEN"))
    print("Logged in to Hugging Face")
except Exception as e:
    print(f"HF login failed (attach the HF_TOKEN secret to this notebook): {e}")
    sys.exit(1)

os.environ.setdefault("HF_USER", "kdt523")
os.makedirs("results", exist_ok=True)
code = sh(f"python {PHASE}")

shutil.copytree("results", f"{WORK}/results", dirs_exist_ok=True)
print(f"{PHASE} exited with code {code}")
sys.exit(code)
