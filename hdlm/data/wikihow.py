import json
import re

from .base import MathDataset


class WikiHow(MathDataset):
    """WikiHow procedural text dataset.

    Expected JSONL format (one JSON object per line):
        {
          "title":   "how to add toolbars to your browsers 1",
          "summary": "launch firefox . open the menu . ...",
          "text":    "double - click on the browsers shortcut icon ..."
        }

    Hierarchy zones:
      - title   (prompt, never masked)     → level 0
      - summary (masked, faster unmask)    → level 0
      - text    (masked, slower unmask)    → level 1

    At inference the model receives the title and generates summary + text.
    Evaluation computes word-overlap F1 on the generated vs gold text (Steps).
    """

    def load_items(self, path: str) -> list:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def format_prompt(self, item) -> str:
        return f"Task: {item['title']}\n"

    def format_full(self, item) -> str:
        return f"Task: {item['title']}\nSummary: {item['summary']}\nSteps: {item['text']}"

    def get_gold_answer(self, item) -> str:
        return item["text"]

    def extract_answer(self, text: str):
        """Return everything after 'Steps:' if present, else the full text."""
        match = re.search(r"Steps:\s*(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip() if text.strip() else None

    def build_hierarchy(self, input_ids: list, tokenizer, item: dict) -> list:
        """Three-zone positional hierarchy.

        title   → level 0 (prompt, never masked)
        summary → level 0 (masked, faster unmask because λ_0 > λ_1)
        text    → level 1 (masked, slower unmask)
        """
        title_end = len(tokenizer(self.format_prompt(item))["input_ids"])
        summary_end = len(
            tokenizer(f"Task: {item['title']}\nSummary: {item['summary']}\n")["input_ids"]
        )
        labels = []
        for i in range(len(input_ids)):
            if i < title_end:
                labels.append(0)       # title: level 0
            elif i < summary_end:
                labels.append(0)       # summary: level 0
            else:
                labels.append(1)       # text: level 1
        return labels

    def answers_match(self, pred_text: str, gold_text: str) -> bool:
        """Word-overlap F1 >= 0.5 between predicted and gold text."""
        pred = self.extract_answer(pred_text)
        gold = self.extract_answer(gold_text)

        if pred is None or gold is None:
            return False

        pred_words = set(pred.lower().split())
        gold_words = set(gold.lower().split())

        if not pred_words or not gold_words:
            return False

        overlap = len(pred_words & gold_words)
        precision = overlap / len(pred_words)
        recall = overlap / len(gold_words)

        if precision + recall == 0:
            return False

        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= 0.5
