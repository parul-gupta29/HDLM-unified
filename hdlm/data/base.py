import re
from abc import ABC, abstractmethod


class MathDataset(ABC):
    """Abstract interface for a Q&A dataset used with HDLM.

    Implement this class to add support for a new dataset, then register it
    in hdlm/data/registry.py so the training and eval scripts can find it.

    Example
    -------
    class MyDataset(MathDataset):
        def load_items(self, path):
            ...
        def format_prompt(self, item):
            return f"Q: {item['input']}"
        def format_full(self, item):
            return f"Q: {item['input']} A: {item['output']}"
        def get_gold_answer(self, item):
            return item['output']
        def extract_answer(self, text):
            ...
        def answers_match(self, pred_text, gold_text):
            ...
    """

    @abstractmethod
    def load_items(self, path: str) -> list:
        """Load raw items from file path (e.g. jsonl, csv)."""

    @abstractmethod
    def format_prompt(self, item) -> str:
        """Return the prompt string shown to the model at inference time.

        Should NOT include the answer — only the question/context.
        """

    @abstractmethod
    def format_full(self, item) -> str:
        """Return the full training string: prompt + answer."""

    @abstractmethod
    def get_gold_answer(self, item) -> str:
        """Return the raw gold answer string for an item."""

    @abstractmethod
    def extract_answer(self, text: str):
        """Extract the final answer from a generated text string.

        Returns the answer as a string, or None if not found.
        """

    @abstractmethod
    def answers_match(self, pred_text: str, gold_text: str) -> bool:
        """Return True if pred_text contains the correct answer vs gold_text."""

    def is_number_token(self, token_text: str) -> bool:
        """Classify a decoded token as hierarchy level 1 (numeric) or 0 (text).

        Used to build hierarchy labels for the HierarchyPredictor.
        Override to customize what counts as a "high-priority" token for
        your dataset (e.g. code tokens, entity mentions, etc.).
        """
        t = token_text.strip()
        return bool(
            re.fullmatch(r"[0-9]+", t)
            or re.fullmatch(r"[0-9\.\+\-\*/=]+", t)
        )

    def build_hierarchy(self, input_ids: list, tokenizer, item: dict) -> list:
        """Return a hierarchy label (int) for each token in input_ids.

        Default: level 1 for numeric tokens, level 0 for all others.
        item is the raw data dict; use it to compute segment boundaries if needed.
        Override this method to implement dataset-specific hierarchy logic.
        """
        return [
            1 if self.is_number_token(tokenizer.decode([tok])) else 0
            for tok in input_ids
        ]
