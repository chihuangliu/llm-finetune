import json
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader, Dataset
from llm_finetune.config import MODEL_NAME
from transformers import AutoTokenizer
from pydantic import BaseModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
default_path = Path(__file__).parent.parent / "data" / "dataset"


class Message(BaseModel):
    role: str
    content: str


class ChatDataset(Dataset):
    def __init__(
        self,
        path: Path | None,
        tokenizer: AutoTokenizer = tokenizer,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.tokenizer = tokenizer
        if not path:
            path = default_path / f"{split}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line)["messages"] for line in f.readlines()]
        self.data = [self._apply_chat_template(messages) for messages in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _apply_chat_template(self, messages: list[Message]):
        return self.tokenizer.apply_chat_template(messages, tokenize=False)


class ChatDataLoader(DataLoader):
    def __init__(
        self,
        dataset: ChatDataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


if __name__ == "__main__":
    dataset = ChatDataset(path=None, split="train")
    dataloader = ChatDataLoader(dataset, batch_size=1, shuffle=False)
    data = next(iter(dataloader))
    print(data[0])
