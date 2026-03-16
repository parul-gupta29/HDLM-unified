"""Train the hierarchical diffusion generator (HierarchicalLLaDA) with LoRA.

Usage:
    python scripts/train_generator.py \
        --dataset gsm8k \
        --train_data /path/to/train.jsonl \
        --model_path /path/to/LLaDA-8B-Instruct \
        --output_dir generator_lora

To add a new dataset, implement hdlm.data.base.MathDataset and register it
in hdlm/data/registry.py, then pass --dataset <name>.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from hdlm.models.generator import HierarchicalLLaDA
from hdlm.models.noise_schedule import HierarchicalNoiseSchedule
from hdlm.data.datasets import GeneratorDataset
from hdlm.data.registry import get_dataset
from hdlm.config import apply_config


def parse_args():
    p = argparse.ArgumentParser(description="Train hierarchical diffusion generator")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (e.g. configs/gsm8k.yaml)")
    p.add_argument("--dataset", default="gsm8k",
                   help="Dataset name (see hdlm/data/registry.py)")
    p.add_argument("--train_data", required=False,
                   help="Path to training data file")
    p.add_argument("--model_path", required=False,
                   help="Path to base LLaDA model")
    apply_config(p, "train_generator")
    p.add_argument("--output_dir", default="generator_lora",
                   help="Where to save LoRA weights")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_levels", type=int, default=2)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--mask_token_id", type=int, default=126336)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def train_step(model, batch, schedule, optimizer, args):
    model.train()

    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)
    hierarchy_labels = batch["hierarchy_labels"].to(args.device)
    prompt_len = batch["prompt_len"]

    B, L = input_ids.shape

    t = torch.randint(1, args.num_timesteps, (B,), device=args.device)
    mask_rates = schedule.get_masking_rate(t, hierarchy_labels)
    mask = torch.rand_like(mask_rates) < mask_rates

    for i in range(B):
        mask[i, :prompt_len[i]] = False

    masked_ids = input_ids.clone()
    masked_ids[mask] = args.mask_token_id

    hier_probs = F.one_hot(hierarchy_labels, num_classes=args.num_levels).float()

    logits = model(masked_ids, hier_probs, attention_mask)

    if mask.sum() == 0:
        return 0.0

    loss = F.cross_entropy(logits[mask], input_ids[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    args = parse_args()

    dataset_cls = get_dataset(args.dataset)
    dataset_spec = dataset_cls()

    dataset = GeneratorDataset(
        args.train_data, args.model_path, dataset_spec,
        max_length=args.max_length
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = HierarchicalLLaDA(
        args.model_path, num_levels=args.num_levels
    ).to(args.device)

    schedule = HierarchicalNoiseSchedule(
        num_levels=args.num_levels, num_timesteps=args.num_timesteps
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            loss = train_step(model, batch, schedule, optimizer, args)
            if step % 20 == 0:
                print(f"Epoch {epoch}  Step {step}  Loss {loss:.4f}")

    model.model.save_pretrained(args.output_dir)
    print(f"Saved LoRA weights to {args.output_dir}")


if __name__ == "__main__":
    main()
