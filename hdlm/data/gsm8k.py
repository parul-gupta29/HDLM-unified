import json
import os
import re

from .base import MathDataset


class GSM8K(MathDataset):
    """GSM8K grade-school math dataset.

    Supports two source formats:
      - Local JSONL file: {"question": "...", "answer": "... #### <number>"}
      - HuggingFace dataset ID with split suffix: "openai/gsm8k:train"
    """

    def load_items(self, path: str) -> list:
        if os.path.isfile(path):
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]

        # HuggingFace dataset: "openai/gsm8k:train" or "openai/gsm8k:test"
        from datasets import load_dataset
        if ":" in path:
            dataset_id, split = path.rsplit(":", 1)
        else:
            dataset_id, split = path, "train"
        ds = load_dataset(dataset_id, "main", split=split)
        return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]

    def format_prompt(self, item) -> str:
        return f"Question: {item['question']}\nAnswer:"

    def format_full(self, item) -> str:
        return f"Question: {item['question']}\nAnswer: {item['answer']}"

    def get_gold_answer(self, item) -> str:
        return item["answer"]

    def extract_answer(self, text: str):
        num_pattern = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

        m = re.search(rf"####\s*({num_pattern})", text)
        if m:
            return m.group(1).replace(",", "")

        nums = re.findall(num_pattern, text)
        return nums[-1].replace(",", "") if nums else None

    def answers_match(self, pred_text: str, gold_text: str) -> bool:
        p = self.extract_answer(pred_text)
        g = self.extract_answer(gold_text)

        if p is None or g is None:
            return False

        try:
            return abs(float(p) - float(g)) < 1e-6
        except ValueError:
            return p.strip() == g.strip()
