import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .base import MathDataset


class GeneratorDataset(Dataset):
    """PyTorch Dataset for training the unified diffusion model.

    Works with any MathDataset implementation. Returns:
        input_ids, attention_mask, prompt_len, hierarchy_labels.
    """

    def __init__(
        self,
        path: str,
        tokenizer_name: str,
        dataset: MathDataset,
        max_length: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.max_length = max_length
        self.dataset = dataset
        self.items = dataset.load_items(path)
        print(f"Loaded {len(self.items)} examples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        prompt = self.dataset.format_prompt(item)
        full_text = self.dataset.format_full(item)

        enc = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        hierarchy = self.dataset.build_hierarchy(input_ids, self.tokenizer, item)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "prompt_len": prompt_len,
            "hierarchy_labels": torch.tensor(hierarchy),
        }


class HierarchyDataset(Dataset):
    """PyTorch Dataset for training the hierarchy predictor.

    Works with any MathDataset implementation. Returns:
        input_ids, attention_mask, hierarchy_labels.

    Hierarchy labels are built by calling dataset.is_number_token() on each
    decoded token — override that method to customize labeling for your task.
    """

    def __init__(
        self,
        path: str,
        tokenizer_name: str,
        dataset: MathDataset,
        max_length: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.max_length = max_length
        self.dataset = dataset
        self.items = dataset.load_items(path)
        print(f"Loaded {len(self.items)} examples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        text = self.dataset.format_full(item)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        hierarchy = self.dataset.build_hierarchy(input_ids, self.tokenizer, item)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "hierarchy_labels": torch.tensor(hierarchy),
        }
