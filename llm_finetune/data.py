from pathlib import Path
from typing import Literal
from datasets import Dataset as HFDataset, load_dataset

from llm_finetune.utils import tokenizer

default_dir = Path(__file__).parent.parent / "data" / "dataset"


def apply_chat_template(batch):
    batch["text"] = tokenizer.apply_chat_template(batch["messages"], tokenize=False)
    return batch


def get_dataset(
    split: Literal["train", "val", "test"], dir: Path = default_dir
) -> HFDataset:
    ds = load_dataset("json", data_files=str(dir / f"{split}.jsonl"), split="train")

    return ds.map(apply_chat_template, batched=True)
