"""
CLI script to generate the synthetic sample dataset.

Usage:
    python scripts/generate_samples.py [--num-per-class N]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.data_engineering.dataset_generator import generate_dataset
import config


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic audio dataset for anti-spoofing training"
    )
    parser.add_argument(
        "--num-per-class", type=int, default=config.NUM_SAMPLES_PER_CLASS,
        help=f"Number of samples per class (default: {config.NUM_SAMPLES_PER_CLASS})"
    )
    args = parser.parse_args()

    print(f"Generating {args.num_per_class} samples per class...")
    result = generate_dataset(num_per_class=args.num_per_class)
    print(f"\nDone! Generated {result['real_count'] + result['spoof_count']} total samples.")


if __name__ == "__main__":
    main()
