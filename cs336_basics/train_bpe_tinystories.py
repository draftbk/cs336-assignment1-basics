#!/usr/bin/env python3
"""
Train a BPE tokenizer on TinyStories dataset.
Vocab size: 10,000 | Special token: <|endoftext|>
"""

from bpe_training_utils import train_bpe_tokenizer

if __name__ == "__main__":
    train_bpe_tokenizer(
        input_path="/home/lifans/cs336/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        output_dir="/home/lifans/cs336/cs336-assignment1-basics/tokenizer_output/tinystories",
        dataset_name="TinyStories",
        time_limit_hours=0.5,  # 30 minutes
        memory_limit_gb=30
    )
