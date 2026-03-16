"""Evaluate the unified hierarchical diffusion model (UnifiedLLaDA) on a Q&A dataset.

Supports best-of-N sampling via --num_samples (default 1).

Usage (with config):
    python scripts/eval_hierarchical.py --config configs/gsm8k.yaml

Usage (explicit args):
    python scripts/eval_hierarchical.py \
        --dataset gsm8k \
        --test_data /path/to/test.jsonl \
        --model_path /path/to/LLaDA-8B-Instruct \
        --unified_model checkpoints/gsm8k/unified_lora

Usage (best-of-4):
    python scripts/eval_hierarchical.py --config configs/gsm8k.yaml --num_samples 4

To add a new dataset, implement hdlm.data.base.MathDataset and register it
in hdlm/data/registry.py, then pass --dataset <name>.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from hdlm.models.generator import UnifiedLLaDA
from hdlm.sampler import HierarchicalSampler
from hdlm.data.registry import get_dataset
from hdlm.config import apply_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate unified hierarchical diffusion model")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (e.g. configs/gsm8k.yaml)")
    p.add_argument("--dataset", default="gsm8k",
                   help="Dataset name (see hdlm/data/registry.py)")
    p.add_argument("--test_data", required=False,
                   help="Path to test data file")
    p.add_argument("--model_path", required=False,
                   help="Path to base LLaDA model")
    apply_config(p, "eval")
    p.add_argument("--unified_model", default="checkpoints/unified_lora",
                   help="Path to trained UnifiedLLaDA LoRA weights")
    p.add_argument("--output", default="hierarchical_results.json")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Samples per question; counts correct if any matches (best-of-N)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--num_levels", type=int, default=2)
    p.add_argument("--mask_token_id", type=int, default=126336)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_cls = get_dataset(args.dataset)
    dataset_spec = dataset_cls()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    print("Loading unified model...")
    model = UnifiedLLaDA(args.model_path, num_levels=args.num_levels)
    model.model = PeftModel.from_pretrained(model.model, args.unified_model)
    model = model.to(args.device).eval()

    print("Building sampler...")
    sampler = HierarchicalSampler(
        model, tokenizer,
        args.mask_token_id, num_levels=args.num_levels,
        num_steps=args.num_steps, device=args.device
    )

    print("Loading test set...")
    items = dataset_spec.load_items(args.test_data)
    print(f"Total examples: {len(items)}")

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
            with torch.no_grad():
                output_ids = sampler.sample(prompt, max_length=args.max_length)

            ids = output_ids[0].tolist()
            prompt_len = len(tokenizer(prompt)["input_ids"])
            full_text = tokenizer.decode(ids, skip_special_tokens=True)
            answer_text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            sample_outputs.append({"full": full_text, "answer": answer_text})

            if dataset_spec.answers_match(answer_text, gold_text):
                success = True

        if dataset_spec.extract_answer(sample_outputs[0]["answer"]) is None:
            no_answer += 1

        if success:
            correct += 1

        entry = {
            "prompt": prompt,
            "gold_answer": gold_text,
            "gold_final": gold_final,
            "correct": success,
        }
        if args.num_samples == 1:
            entry["full_text"] = sample_outputs[0]["full"]
            entry["answer_text"] = sample_outputs[0]["answer"]
            entry["pred_final"] = dataset_spec.extract_answer(sample_outputs[0]["answer"])
        else:
            entry["samples"] = sample_outputs

        results.append(entry)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(items)}]  Accuracy: {correct / (i + 1):.3f}")

    total = len(items)
    accuracy = correct / total * 100

    print("\n" + "=" * 50)
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
            "num_samples": args.num_samples,
            "dataset": args.dataset,
            "unified_model": args.unified_model,
            "examples": results,
        }, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
