"""Dataset registry.

To add a new dataset:
1. Create a class in hdlm/data/ that subclasses MathDataset.
2. Add it to the DATASETS dict below.
3. Pass --dataset <name> to any training or eval script.
"""

from .gsm8k import GSM8K
from .wikihow import WikiHow

DATASETS = {
    "gsm8k": GSM8K,
    "wikihow": WikiHow,
}


def get_dataset(name: str) -> type:
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )
    return DATASETS[name]
