from llm_finetune.config import MODEL_NAME
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def _get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


device = _get_device()
