"""Kaggle runner: pull the latest code from GitHub and run one phase.

Pushed from the laptop with:
  python -m kaggle kernels push -p kaggle/phase0 --accelerator NvidiaTeslaT4
Outputs land in /kaggle/working/results (downloaded with `kaggle kernels output`).
"""
import os
import shutil
import subprocess
import sys

PHASE = "phase1_build_dataset.py"
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

from huggingface_hub import login

# Secrets are not available to runs started from the CLI, so the token comes
# from a private dataset (krushnad523/hf-token); the secret is a fallback for UI runs.
TOKEN_FILE = "/kaggle/input/hf-token/hf_token.txt"
try:
    if os.path.exists(TOKEN_FILE):
        token = open(TOKEN_FILE).read().strip()
    else:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    login(token)
    print("Logged in to Hugging Face")
except Exception as e:
    print(f"HF login failed (add the private hf-token dataset): {e}")
    sys.exit(1)

os.environ.setdefault("HF_USER", "kd1609")
os.makedirs("results", exist_ok=True)
code = sh(f"python {PHASE}")

for d in ("results", "data"):
    if os.path.isdir(d):
        shutil.copytree(d, f"{WORK}/{d}", dirs_exist_ok=True)
print(f"{PHASE} exited with code {code}")
sys.exit(code)
