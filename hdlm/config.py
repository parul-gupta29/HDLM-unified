"""Config file loading utilities.

YAML config files are organised into sections that map to individual scripts.
CLI arguments always take precedence over config file values.

Config structure
----------------
dataset: <name>
data:
  train: <path>
  test:  <path>
model:
  path: <path>
  mask_token_id: <int>
  num_levels: <int>
  max_length: <int>
train_hierarchy:   <script-specific overrides>
train_generator:   <script-specific overrides>
eval:              <script-specific overrides>
"""

import yaml


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _flatten(config: dict, section: str) -> dict:
    """Build a flat dict of argparse-compatible defaults from a config dict.

    Args:
        config:  Parsed YAML dict.
        section: Which script section to pull (e.g. 'train_generator', 'eval').

    Returns:
        Flat dict mapping arg names to values (Nones excluded).
    """
    flat = {}

    if "dataset" in config:
        flat["dataset"] = config["dataset"]

    if "data" in config:
        flat["train_data"] = config["data"].get("train")
        flat["test_data"] = config["data"].get("test")

    if "model" in config:
        m = config["model"]
        # Both --model_path (train/eval_hierarchical) and
        # --base_model (eval_baseline) map to model.path
        flat["model_path"] = m.get("path")
        flat["base_model"] = m.get("path")
        flat["mask_token_id"] = m.get("mask_token_id")
        flat["num_levels"] = m.get("num_levels")
        flat["max_length"] = m.get("max_length")

    if section and section in config:
        flat.update(config[section])

    return {k: v for k, v in flat.items() if v is not None}


def apply_config(parser, section: str):
    """Check for --config in sys.argv, load it, and set argparse defaults.

    Call this inside parse_args() BEFORE parser.parse_args():

        parser.add_argument('--config', default=None)
        apply_config(parser, 'train_generator')
        return parser.parse_args()

    Args:
        parser:  ArgumentParser instance (must already have --config added).
        section: Config section name for this script.
    """
    import sys

    # Quick pre-scan for --config without a full parse (avoids required-arg errors)
    config_path = None
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in ("--config", "-c") and i + 1 < len(argv):
            candidate = argv[i + 1]
            if not candidate.startswith("-"):
                config_path = candidate
            break
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            break

    if config_path is None:
        return

    config = load_yaml(config_path)
    defaults = _flatten(config, section)
    parser.set_defaults(**defaults)
