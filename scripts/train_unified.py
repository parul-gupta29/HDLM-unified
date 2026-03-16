"""Train the unified LLaDA model with joint hierarchy and LM objectives.

A single shared backbone serves both the hierarchy head (Pass 1) and the
LM head (Pass 2).  Gradients from both losses flow through shared weights.

Usage:
    python scripts/train_unified.py \\
        --config configs/gsm8k.yaml \\
        --train_data /path/to/train.jsonl \\
        --model_path /path/to/LLaDA-8B-Instruct

To add a new dataset, implement hdlm.data.base.MathDataset and register it
in hdlm/data/registry.py, then pass --dataset <name>.
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from hdlm.models.generator import UnifiedLLaDA
from hdlm.models.noise_schedule import HierarchicalNoiseSchedule
from hdlm.data.datasets import GeneratorDataset
from hdlm.data.registry import get_dataset
from hdlm.config import apply_config


def parse_args():
    p = argparse.ArgumentParser(description="Train unified hierarchical diffusion model")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (e.g. configs/gsm8k.yaml)")
    p.add_argument("--dataset", default="gsm8k",
                   help="Dataset name (see hdlm/data/registry.py)")
    p.add_argument("--train_data", required=False,
                   help="Path to training data file")
    p.add_argument("--model_path", required=False,
                   help="Path to base LLaDA model")
    apply_config(p, "train_unified")
    p.add_argument("--output_dir", default="checkpoints/unified_lora",
                   help="Where to save model weights")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_levels", type=int, default=2)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--mask_token_id", type=int, default=126336)
    p.add_argument("--hier_loss_weight", type=float, default=0.3,
                   help="Weight λ for hierarchy loss: L = L_lm + λ * L_hier")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of training data to use for validation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb_project", default="hdlm",
                   help="Weights & Biases project name (set to 'disabled' to turn off)")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (defaults to auto-generated)")
    return p.parse_args()


def train_step(model, batch, schedule, args):
    """Run one micro-batch forward+backward. Returns losses (not yet stepped)."""
    model.train()

    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)
    hierarchy_labels = batch["hierarchy_labels"].to(args.device)
    prompt_len = batch["prompt_len"]

    B, L = input_ids.shape

    # --- Sample timestep and compute mask ---
    t = torch.randint(1, args.num_timesteps, (B,), device=args.device)
    mask_rates = schedule.get_masking_rate(t, hierarchy_labels)
    mask = torch.rand_like(mask_rates) < mask_rates

    for i in range(B):
        mask[i, :prompt_len[i]] = False

    if mask.sum() == 0:
        return None, None, None

    masked_ids = input_ids.clone()
    masked_ids[mask] = args.mask_token_id

    # --- Pass 1 context: W[gold_label] for unmasked, mean(W) for masked ---
    hier_context_embeds = model.build_hier_context_embeds(hierarchy_labels, mask)

    # --- Combined forward: Pass 1 (hierarchy) + Pass 2 (LM) ---
    lm_logits, hier_logits, _ = model(masked_ids, hier_context_embeds, attention_mask)

    # --- LM loss: cross-entropy on masked positions ---
    L_lm = F.cross_entropy(lm_logits[mask], input_ids[mask])

    # --- Hierarchy loss: cross-entropy on masked answer positions only ---
    # Unmasked positions have gold labels injected as context in Pass 1,
    # so predicting them back is trivial. The hierarchy head only needs to
    # learn to predict levels for masked positions where context = mean(W).
    answer_mask = torch.zeros(B, L, dtype=torch.bool, device=args.device)
    for i in range(B):
        answer_mask[i, prompt_len[i]:] = attention_mask[i, prompt_len[i]:].bool()
    hier_mask = answer_mask & mask

    if hier_mask.any():
        L_hier = F.cross_entropy(
            hier_logits[hier_mask],
            hierarchy_labels[hier_mask]
        )
    else:
        L_hier = torch.tensor(0.0, device=args.device)

    loss = (L_lm + args.hier_loss_weight * L_hier) / args.grad_accum_steps
    loss.backward()

    return loss.item() * args.grad_accum_steps, L_lm.item(), L_hier.item()


@torch.no_grad()
def val_epoch(model, loader, schedule, args):
    """Compute average val losses over the full validation set."""
    model.eval()
    total_loss = total_lm = total_hier = 0.0
    count = 0

    for batch in loader:
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

        if mask.sum() == 0:
            continue

        masked_ids = input_ids.clone()
        masked_ids[mask] = args.mask_token_id

        hier_context_embeds = model.build_hier_context_embeds(hierarchy_labels, mask)
        lm_logits, hier_logits, _ = model(masked_ids, hier_context_embeds, attention_mask)

        L_lm = F.cross_entropy(lm_logits[mask], input_ids[mask])

        answer_mask = torch.zeros(B, L, dtype=torch.bool, device=args.device)
        for i in range(B):
            answer_mask[i, prompt_len[i]:] = attention_mask[i, prompt_len[i]:].bool()
        hier_mask = answer_mask & mask

        L_hier = (
            F.cross_entropy(hier_logits[hier_mask], hierarchy_labels[hier_mask])
            if hier_mask.any()
            else torch.tensor(0.0, device=args.device)
        )

        total_lm += L_lm.item()
        total_hier += L_hier.item()
        total_loss += (L_lm + args.hier_loss_weight * L_hier).item()
        count += 1

    if count == 0:
        return None
    return {
        "val/loss": total_loss / count,
        "val/lm_loss": total_lm / count,
        "val/hier_loss": total_hier / count,
    }


def main():
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join("checkpoints/gsm8k/unified", run_id)

    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "dataset": args.dataset,
            "model_path": args.model_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "lr": args.lr,
            "max_length": args.max_length,
            "num_levels": args.num_levels,
            "num_timesteps": args.num_timesteps,
            "hier_loss_weight": args.hier_loss_weight,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        mode="disabled" if args.wandb_project == "disabled" else "online",
    )

    dataset_cls = get_dataset(args.dataset)
    dataset_spec = dataset_cls()

    full_dataset = GeneratorDataset(
        args.train_data, args.model_path, dataset_spec,
        max_length=args.max_length
    )
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}  Val: {val_size}")
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UnifiedLLaDA(
        args.model_path, num_levels=args.num_levels,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    ).to(args.device)
    model.model.print_trainable_parameters()

    schedule = HierarchicalNoiseSchedule(
        num_levels=args.num_levels, num_timesteps=args.num_timesteps
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        accum_loss = accum_lm = accum_hier = 0.0
        accum_count = 0

        for step, batch in enumerate(loader):
            loss, lm_loss, hier_loss = train_step(model, batch, schedule, args)

            if loss is None:   # all-unmasked batch, skip
                continue

            accum_loss += loss
            accum_lm += lm_loss
            accum_hier += hier_loss
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({
                    "loss": accum_loss / args.grad_accum_steps,
                    "lm_loss": accum_lm / args.grad_accum_steps,
                    "hier_loss": accum_hier / args.grad_accum_steps,
                }, step=global_step)

                if global_step % 20 == 0:
                    print(
                        f"Epoch {epoch}  Step {global_step}  "
                        f"Loss {accum_loss / args.grad_accum_steps:.4f}  "
                        f"LM {accum_lm / args.grad_accum_steps:.4f}  "
                        f"Hier {accum_hier / args.grad_accum_steps:.4f}"
                    )

                accum_loss = accum_lm = accum_hier = 0.0

        # Save per-epoch checkpoint
        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.model.save_pretrained(ckpt_dir)
        torch.save(
            {
                "hierarchy_head": model.hierarchy_head.state_dict(),
                "hierarchy_embedding": model.hierarchy_embedding.state_dict(),
            },
            os.path.join(ckpt_dir, "hierarchy_components.pt"),
        )
        print(f"Saved epoch {epoch} checkpoint to {ckpt_dir}")

        # Validation
        val_metrics = val_epoch(model, val_loader, schedule, args)
        if val_metrics:
            wandb.log(val_metrics, step=global_step)
            print(
                f"Val epoch {epoch}  "
                f"Loss {val_metrics['val/loss']:.4f}  "
                f"LM {val_metrics['val/lm_loss']:.4f}  "
                f"Hier {val_metrics['val/hier_loss']:.4f}"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save LoRA weights
    model.model.save_pretrained(args.output_dir)

    # Save hierarchy head and embedding matrix separately
    torch.save(
        {
            "hierarchy_head": model.hierarchy_head.state_dict(),
            "hierarchy_embedding": model.hierarchy_embedding.state_dict(),
        },
        os.path.join(args.output_dir, "hierarchy_components.pt"),
    )

    print(f"Saved unified model to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
