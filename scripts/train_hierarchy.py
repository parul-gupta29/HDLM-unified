"""Train the hierarchy predictor (token-level classifier).

Usage:
    python scripts/train_hierarchy.py \
        --dataset gsm8k \
        --train_data /path/to/train.jsonl \
        --model_path /path/to/LLaDA-8B-Instruct \
        --output hierarchy_predictor.pt

To add a new dataset, implement hdlm.data.base.MathDataset and register it
in hdlm/data/registry.py, then pass --dataset <name>.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hdlm.models.hierarchy_predictor import HierarchyPredictor
from hdlm.models.noise_schedule import HierarchicalNoiseSchedule
from hdlm.data.datasets import HierarchyDataset
from hdlm.data.registry import get_dataset
from hdlm.config import apply_config


def parse_args():
    p = argparse.ArgumentParser(description="Train hierarchy predictor")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (e.g. configs/gsm8k.yaml)")
    p.add_argument("--dataset", default="gsm8k",
                   help="Dataset name (see hdlm/data/registry.py)")
    p.add_argument("--train_data", required=False,
                   help="Path to training data file")
    p.add_argument("--model_path", required=False,
                   help="Path to base LLaDA model (for tokenizer)")
    apply_config(p, "train_hierarchy")
    p.add_argument("--output", default="hierarchy_predictor.pt",
                   help="Where to save the trained model")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_levels", type=int, default=2)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def train_step(model, batch, schedule, optimizer, args):
    model.train()

    input_ids = batch["input_ids"].to(args.device)
    hierarchy_labels = batch["hierarchy_labels"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)

    B = input_ids.shape[0]

    t = torch.randint(1, args.num_timesteps, (B,), device=args.device)
    keep_mask = schedule.sample_masks(t, hierarchy_labels)
    mask = 1 - keep_mask.long()

    masked_ids = input_ids.clone()
    masked_ids[mask.bool()] = 126336

    hierarchy_input = hierarchy_labels.clone()
    hierarchy_input[mask.bool()] = args.num_levels

    logits = model(masked_ids, hierarchy_input, attention_mask)

    loss = model.compute_loss(logits, hierarchy_labels, mask)

    pred = logits.argmax(-1)
    correct = (pred[mask.bool()] == hierarchy_labels[mask.bool()]).float()
    acc = correct.mean() if correct.numel() > 0 else torch.tensor(0.0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    dataset_cls = get_dataset(args.dataset)
    dataset_spec = dataset_cls()

    dataset = HierarchyDataset(
        args.train_data, args.model_path, dataset_spec,
        max_length=args.max_length
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = HierarchyPredictor(
        vocab_size=len(tokenizer), num_levels=args.num_levels
    ).to(args.device)

    schedule = HierarchicalNoiseSchedule(
        num_levels=args.num_levels, num_timesteps=args.num_timesteps
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            loss, acc = train_step(model, batch, schedule, optimizer, args)
            if step % 50 == 0:
                print(f"Epoch {epoch}  Step {step}  Loss {loss:.4f}  Acc {acc:.3f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved hierarchy predictor to {args.output}")


if __name__ == "__main__":
    main()
