"""Evaluate vanilla LLaDA diffusion (no hierarchy) on a Q&A dataset.

Supports best-of-N sampling via --num_samples (default 1).

Usage (single sample):
    python scripts/eval_baseline.py \
        --dataset gsm8k \
        --test_data /path/to/test.jsonl \
        --base_model /path/to/LLaDA-8B-Instruct

Usage (with LoRA, best-of-4):
    python scripts/eval_baseline.py ... \
        --lora_path /path/to/lora_weights \
        --num_samples 4

To add a new dataset, implement hdlm.data.base.MathDataset and register it
in hdlm/data/registry.py, then pass --dataset <name>.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from hdlm.data.registry import get_dataset
from hdlm.config import apply_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline LLaDA diffusion model")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (e.g. configs/gsm8k.yaml)")
    p.add_argument("--dataset", default="gsm8k",
                   help="Dataset name (see hdlm/data/registry.py)")
    p.add_argument("--test_data", required=False,
                   help="Path to test data file")
    p.add_argument("--base_model", required=False,
                   help="Path to base LLaDA model")
    apply_config(p, "eval")
    p.add_argument("--lora_path", default=None,
                   help="Path to LoRA weights (omit to use base model as-is)")
    p.add_argument("--output", default="baseline_results.json")
    p.add_argument("--steps", type=int, default=128,
                   help="Number of diffusion denoising steps")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Samples per question; counts correct if any matches (best-of-N)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size (only used when num_samples=1)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_examples", type=int, default=None,
                   help="Cap on number of test examples (for debugging)")
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def llada_inpaint(model, input_ids, prefix_lens, mask_id, num_steps, device):
    """Full-sequence masked diffusion inpainting.

    Question tokens are pinned; answer tokens are progressively unmasked
    using LLaDA's cosine unmasking schedule.
    """
    B, L = input_ids.shape
    x = input_ids.clone().to(device)

    eps = 1e-5
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

    for i in range(num_steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        logits = model(x).logits
        x0_pred = logits.argmax(dim=-1)

        unmask_prob = (t_now - t_next) / t_now
        is_masked = (x == mask_id)
        unmask_flag = (torch.rand(B, L, device=device) < unmask_prob) & is_masked

        x_new = x.clone()
        x_new[unmask_flag] = x0_pred[unmask_flag]

        for b in range(B):
            plen = prefix_lens[b].item()
            x_new[b, :plen] = input_ids[b, :plen]

        x = x_new

    if (x == mask_id).any():
        x0_final = model(x).logits.argmax(dim=-1)
        still_masked = (x == mask_id)
        x[still_masked] = x0_final[still_masked]
        for b in range(B):
            x[b, :prefix_lens[b].item()] = input_ids[b, :prefix_lens[b].item()]

    return x.cpu()


def run_single_sample(
    model, tokenizer, items, dataset_spec, mask_id, args
):
    """Batched evaluation with a single sample per question."""
    correct = 0
    no_answer = 0
    results = []
    total = len(items)

    for i in range(0, total, args.batch_size):
        batch = items[i: i + args.batch_size]
        bsize = len(batch)

        prompts = [dataset_spec.format_prompt(ex) for ex in batch]
        gold_texts = [dataset_spec.get_gold_answer(ex) for ex in batch]

        enc = tokenizer(
            prompts,
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        prefix_ids = enc["input_ids"].to(args.device)
        prefix_lens = enc["attention_mask"].sum(dim=1)

        input_ids = torch.full(
            (bsize, args.max_length), mask_id,
            dtype=torch.long, device=args.device
        )
        for b in range(bsize):
            plen = prefix_lens[b].item()
            input_ids[b, :plen] = prefix_ids[b, :plen]

        out = llada_inpaint(
            model, input_ids, prefix_lens, mask_id, args.steps, args.device
        )

        for b in range(bsize):
            ids = out[b].tolist()
            plen = prefix_lens[b].item()
            full_text = tokenizer.decode(ids, skip_special_tokens=True)
            ans_text = tokenizer.decode(ids[plen:], skip_special_tokens=True)

            pred_final = dataset_spec.extract_answer(ans_text)
            gold_final = dataset_spec.extract_answer(gold_texts[b])
            match = dataset_spec.answers_match(ans_text, gold_texts[b])

            if pred_final is None:
                no_answer += 1
            if match:
                correct += 1

            results.append({
                "prompt": prompts[b],
                "gold_answer": gold_texts[b],
                "gold_final": gold_final,
                "full_text": full_text,
                "answer_text": ans_text,
                "pred_final": pred_final,
                "correct": match,
            })

        print(f"  [{min(i + args.batch_size, total)}/{total}]", end="\r")

    return results, correct, no_answer


def run_multi_sample(
    model, tokenizer, items, dataset_spec, mask_id, args
):
    """Per-example best-of-N evaluation."""
    correct = 0
    no_answer = 0
    results = []

    for i, item in enumerate(items):
        prompt = dataset_spec.format_prompt(item)
        gold_text = dataset_spec.get_gold_answer(item)
        gold_final = dataset_spec.extract_answer(gold_text)

        success = False
        sample_outputs = []

        for _ in range(args.num_samples):
            enc = tokenizer(
                prompt,
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            )
            prefix_ids = enc["input_ids"].to(args.device)
            prefix_len = enc["attention_mask"].sum()

            input_ids = torch.full(
                (1, args.max_length), mask_id,
                dtype=torch.long, device=args.device
            )
            input_ids[0, :prefix_len] = prefix_ids[0, :prefix_len]

            out = llada_inpaint(
                model, input_ids,
                prefix_len.unsqueeze(0),
                mask_id, args.steps, args.device
            )

            ids = out[0].tolist()
            full_text = tokenizer.decode(ids, skip_special_tokens=True)
            sample_outputs.append(full_text)

            if dataset_spec.answers_match(full_text, gold_text):
                success = True

        if dataset_spec.extract_answer(sample_outputs[0]) is None:
            no_answer += 1
        if success:
            correct += 1

        results.append({
            "prompt": prompt,
            "gold_answer": gold_text,
            "gold_final": gold_final,
            "samples": sample_outputs,
            "correct": success,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(items)}]  Accuracy: {correct / (i + 1):.3f}")

    return results, correct, no_answer


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device} | Steps: {args.steps} | Samples: {args.num_samples}")

    dataset_cls = get_dataset(args.dataset)
    dataset_spec = dataset_cls()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    mask_id = tokenizer.added_tokens_encoder.get("<|mdm_mask|>", 126336)
    print(f"mask_id={mask_id}")

    items = dataset_spec.load_items(args.test_data)
    if args.max_examples:
        items = items[:args.max_examples]
    print(f"Loaded {len(items)} test examples")

    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)

    if args.lora_path:
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        print(f"Loaded LoRA from {args.lora_path}")
    else:
        model = base_model
        print("Using base model (no LoRA)")

    model.eval()

    if args.num_samples > 1:
        results, correct, no_answer = run_multi_sample(
            model, tokenizer, items, dataset_spec, mask_id, args
        )
    else:
        results, correct, no_answer = run_single_sample(
            model, tokenizer, items, dataset_spec, mask_id, args
        )

    total = len(items)
    accuracy = correct / total * 100

    print()
    print("=" * 50)
    print(f"Accuracy: {accuracy:.2f}%  ({correct}/{total})")
    print(f"No answer: {no_answer}/{total}")
    if args.num_samples > 1:
        print(f"Sampling: best-of-{args.num_samples}")
    print("=" * 50)

    with open(args.output, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "no_answer": no_answer,
            "steps": args.steps,
            "num_samples": args.num_samples,
            "dataset": args.dataset,
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "examples": results,
        }, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
