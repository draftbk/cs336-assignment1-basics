#!/usr/bin/env python3
"""
Compare tokenizers trained on TinyStories vs OpenWebText.

This script answers question (b):
"Compare and contrast the tokenizer that you get training on TinyStories versus
OpenWebText."
"""

import json
from pathlib import Path
from collections import Counter

def load_tokenizer(output_dir, prefix):
    """Load vocab and merges from saved files."""
    output_dir = Path(output_dir)

    # Load raw vocab
    with open(output_dir / f"{prefix}vocab_raw.json") as f:
        vocab_hex = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_hex.items()}

    # Load raw merges
    merges = []
    with open(output_dir / f"{prefix}merges_raw.txt") as f:
        for line in f:
            if line.strip():
                token1_hex, token2_hex = line.strip().split()
                merges.append((bytes.fromhex(token1_hex), bytes.fromhex(token2_hex)))

    return vocab, merges


def analyze_tokenizer_characteristics(vocab, merges, name):
    """Analyze characteristics of a tokenizer."""
    print(f"\n{'='*70}")
    print(f"{name.upper()} TOKENIZER CHARACTERISTICS")
    print(f"{'='*70}")

    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Number of merges: {len(merges)}")

    # Token length distribution
    token_lengths = [len(v) for v in vocab.values()]
    print(f"\nToken Length Distribution:")
    print(f"  Min: {min(token_lengths)} bytes")
    print(f"  Max: {max(token_lengths)} bytes")
    print(f"  Average: {sum(token_lengths)/len(token_lengths):.2f} bytes")

    # Longest tokens
    longest_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"\nTop 5 Longest Tokens:")
    for token_id, token_bytes in longest_tokens:
        try:
            decoded = token_bytes.decode('utf-8')
            print(f"  {len(token_bytes)} bytes: '{decoded}'")
        except UnicodeDecodeError:
            print(f"  {len(token_bytes)} bytes: [non-UTF-8] {token_bytes.hex()[:40]}...")

    # Character type analysis (approximate - decode what we can)
    alphanumeric_count = 0
    special_char_count = 0
    multi_word_count = 0

    for token_bytes in vocab.values():
        try:
            decoded = token_bytes.decode('utf-8')
            if decoded.isalnum():
                alphanumeric_count += 1
            elif not decoded.isspace():
                special_char_count += 1
            if ' ' in decoded and len(decoded.split()) > 1:
                multi_word_count += 1
        except UnicodeDecodeError:
            pass

    print(f"\nToken Type Distribution (approximate):")
    print(f"  Alphanumeric tokens: {alphanumeric_count} ({alphanumeric_count/len(vocab)*100:.1f}%)")
    print(f"  Special character tokens: {special_char_count} ({special_char_count/len(vocab)*100:.1f}%)")
    print(f"  Multi-word tokens: {multi_word_count} ({multi_word_count/len(vocab)*100:.1f}%)")

    # First and last merges
    print(f"\nFirst 5 Merges (most common patterns):")
    for i, (t1, t2) in enumerate(merges[:5], 1):
        try:
            print(f"  {i}. '{t1.decode('utf-8')}' + '{t2.decode('utf-8')}' -> '{(t1+t2).decode('utf-8')}'")
        except UnicodeDecodeError:
            print(f"  {i}. {t1.hex()} + {t2.hex()}")

    print(f"\nLast 5 Merges (least common patterns):")
    for i, (t1, t2) in enumerate(merges[-5:], len(merges)-4):
        try:
            print(f"  {i}. '{t1.decode('utf-8')}' + '{t2.decode('utf-8')}' -> '{(t1+t2).decode('utf-8')}'")
        except UnicodeDecodeError:
            print(f"  {i}. {t1.hex()} + {t2.hex()}")

    return {
        'vocab_size': len(vocab),
        'num_merges': len(merges),
        'avg_token_length': sum(token_lengths)/len(token_lengths),
        'max_token_length': max(token_lengths),
        'alphanumeric_pct': alphanumeric_count/len(vocab)*100,
        'multi_word_count': multi_word_count
    }


def compare_tokenizers(tinystories_stats, owt_stats):
    """Compare two tokenizers and print analysis."""
    print(f"\n{'='*70}")
    print("COMPARISON: TINYSTORIES vs OPENWEBTEXT")
    print(f"{'='*70}")

    print(f"\nVocabulary Size:")
    print(f"  TinyStories: {tinystories_stats['vocab_size']}")
    print(f"  OpenWebText: {owt_stats['vocab_size']}")
    print(f"  Difference: {owt_stats['vocab_size'] - tinystories_stats['vocab_size']}")

    print(f"\nAverage Token Length:")
    print(f"  TinyStories: {tinystories_stats['avg_token_length']:.2f} bytes")
    print(f"  OpenWebText: {owt_stats['avg_token_length']:.2f} bytes")
    print(f"  {'OpenWebText tokens are longer' if owt_stats['avg_token_length'] > tinystories_stats['avg_token_length'] else 'TinyStories tokens are longer'}")

    print(f"\nMaximum Token Length:")
    print(f"  TinyStories: {tinystories_stats['max_token_length']} bytes")
    print(f"  OpenWebText: {owt_stats['max_token_length']} bytes")

    print(f"\nAlphanumeric Token Percentage:")
    print(f"  TinyStories: {tinystories_stats['alphanumeric_pct']:.1f}%")
    print(f"  OpenWebText: {owt_stats['alphanumeric_pct']:.1f}%")

    print(f"\nMulti-word Tokens:")
    print(f"  TinyStories: {tinystories_stats['multi_word_count']}")
    print(f"  OpenWebText: {owt_stats['multi_word_count']}")

    print(f"\n{'='*70}")
    print("ANSWER TO QUESTION (b)")
    print(f"{'='*70}")
    print("\nKey Differences:")
    print("1. [Vocabulary size and coverage]: ")
    print("   TODO: Add analysis about how vocab size affects coverage")
    print("\n2. [Token characteristics and domain specificity]: ")
    print("   TODO: Add analysis about domain-specific patterns")
    print("\nOne-to-two sentence summary:")
    print("TODO: Write 1-2 sentences comparing the tokenizers based on above analysis")


if __name__ == "__main__":
    base_dir = Path("/home/lifans/cs336/cs336-assignment1-basics/tokenizer_output")

    print("="*70)
    print("TOKENIZER COMPARISON TOOL")
    print("="*70)
    print("\nLoading tokenizers...")

    try:
        tinystories_vocab, tinystories_merges = load_tokenizer(base_dir / "tinystories", "tinystories_")
        print("✓ Loaded TinyStories tokenizer")
    except FileNotFoundError:
        print("✗ TinyStories tokenizer not found. Run train_bpe_tinystories.py first.")
        exit(1)

    try:
        owt_vocab, owt_merges = load_tokenizer(base_dir / "openwebtext", "openwebtext_")
        print("✓ Loaded OpenWebText tokenizer")
    except FileNotFoundError:
        print("✗ OpenWebText tokenizer not found. Run train_bpe_expts_owt.py first.")
        exit(1)

    # Analyze each tokenizer
    tinystories_stats = analyze_tokenizer_characteristics(
        tinystories_vocab, tinystories_merges, "TinyStories"
    )

    owt_stats = analyze_tokenizer_characteristics(
        owt_vocab, owt_merges, "OpenWebText"
    )

    # Compare them
    compare_tokenizers(tinystories_stats, owt_stats)
