from pathlib import Path
import subprocess

from llm_finetune.config import MODEL_NAME
from transformers import AutoTokenizer
import torch
from llm_finetune.template_mask import add_assistant_mask
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
add_assistant_mask(tokenizer)


def _get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


device = _get_device()


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
