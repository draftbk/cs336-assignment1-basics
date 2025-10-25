#!/usr/bin/env python3
"""
Train a BPE tokenizer on OpenWebText dataset.
Vocab size: 32,000 | Special token: <|endoftext|>

This script answers question (a):
"Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum
vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk
for further inspection. What is the longest token in the vocabulary? Does it make sense?"

Resource requirements: ≤ 12 hours, ≤ 100GB RAM
"""

from bpe_training_utils import train_bpe_tokenizer

if __name__ == "__main__":
    vocab, merges, time_taken, memory_used, longest_token = train_bpe_tokenizer(
        input_path="/home/lifans/cs336/data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        output_dir="/home/lifans/cs336/cs336-assignment1-basics/tokenizer_output/openwebtext",
        dataset_name="OpenWebText",
        time_limit_hours=12,
        memory_limit_gb=100
    )

    print("\n" + "="*70)
    print("ANSWER TO QUESTION (a)")
    print("="*70)
    print(f"\nLongest token in vocabulary: '{longest_token}'")
    print(f"Length: {len(longest_token.encode('utf-8'))} bytes")
    print("\nDoes it make sense? ")
    print("TODO: Add your 1-2 sentence analysis here after inspecting the token.")
