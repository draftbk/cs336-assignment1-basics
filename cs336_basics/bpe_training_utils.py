#!/usr/bin/env python3
"""
Shared utility functions for BPE tokenizer training experiments.
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.adapters import run_train_bpe
from tests.common import gpt2_bytes_to_unicode


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.2f} hours ({minutes:.2f} minutes)"


def format_bytes(bytes_size):
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def analyze_vocab(vocab, merges, show_merges=True):
    """Analyze the trained vocabulary and merges."""
    print("\n" + "="*70)
    print("VOCABULARY ANALYSIS")
    print("="*70)

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    print(f"\nLongest token:")
    print(f"  Token ID: {longest_token_id}")
    print(f"  Length: {len(longest_token)} bytes")
    print(f"  Bytes (hex): {longest_token.hex()}")
    try:
        decoded = longest_token.decode('utf-8')
        print(f"  Decoded: '{decoded}'")
        # Return the decoded string for use in answers
        return decoded
    except UnicodeDecodeError:
        print(f"  Decoded: [Cannot decode as UTF-8]")
        return None
    finally:
        # Show token length statistics
        token_lengths = [len(v) for v in vocab.values()]
        avg_length = sum(token_lengths) / len(token_lengths)

        print(f"\nToken length statistics:")
        print(f"  Average: {avg_length:.2f} bytes")
        print(f"  Min: {min(token_lengths)} bytes")
        print(f"  Max: {max(token_lengths)} bytes")

        if show_merges:
            # Show first few merges
            print(f"\nFirst 10 merges:")
            for i, (token1, token2) in enumerate(merges[:10]):
                try:
                    token1_str = token1.decode('utf-8')
                    token2_str = token2.decode('utf-8')
                    merged_str = (token1 + token2).decode('utf-8')
                    print(f"  {i+1}. '{token1_str}' + '{token2_str}' -> '{merged_str}'")
                except UnicodeDecodeError:
                    print(f"  {i+1}. {token1.hex()} + {token2.hex()} -> {(token1+token2).hex()}")

            # Show last few merges
            print(f"\nLast 10 merges:")
            for i, (token1, token2) in enumerate(merges[-10:], start=len(merges)-9):
                try:
                    token1_str = token1.decode('utf-8')
                    token2_str = token2.decode('utf-8')
                    merged_str = (token1 + token2).decode('utf-8')
                    print(f"  {i}. '{token1_str}' + '{token2_str}' -> '{merged_str}'")
                except UnicodeDecodeError:
                    print(f"  {i}. {token1.hex()} + {token2.hex()} -> {(token1+token2).hex()}")


def save_vocab_and_merges(vocab, merges, output_dir, name_prefix=""):
    """Save vocabulary and merges to disk in GPT-2 format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert vocab to GPT-2 format (token_id -> string)
    # Using token_id as key to avoid collisions when multiple token_bytes map to same string
    byte_encoder = gpt2_bytes_to_unicode()

    # Create vocab.json (token_id -> token_string)
    vocab_gpt2 = {}
    for token_id, token_bytes in vocab.items():
        # Convert bytes to GPT-2 string representation
        token_str = ''.join(byte_encoder[b] for b in token_bytes)
        vocab_gpt2[str(token_id)] = token_str  # Use token_id as key to prevent collisions

    vocab_path = output_dir / f"{name_prefix}vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_gpt2, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to: {vocab_path}")

    # Create merges.txt (GPT-2 format)
    merges_path = output_dir / f"{name_prefix}merges.txt"
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.write("#version: 0.2\n")  # GPT-2 format header
        for token1, token2 in merges:
            token1_str = ''.join(byte_encoder[b] for b in token1)
            token2_str = ''.join(byte_encoder[b] for b in token2)
            f.write(f"{token1_str} {token2_str}\n")
    print(f"Saved merges to: {merges_path}")

    # Also save raw Python format for easy inspection
    raw_vocab_path = output_dir / f"{name_prefix}vocab_raw.json"
    with open(raw_vocab_path, 'w') as f:
        # Convert bytes to hex strings for JSON serialization
        vocab_serializable = {k: v.hex() for k, v in vocab.items()}
        json.dump(vocab_serializable, f, indent=2)
    print(f"Saved raw vocabulary to: {raw_vocab_path}")

    raw_merges_path = output_dir / f"{name_prefix}merges_raw.txt"
    with open(raw_merges_path, 'w') as f:
        for token1, token2 in merges:
            f.write(f"{token1.hex()} {token2.hex()}\n")
    print(f"Saved raw merges to: {raw_merges_path}")


def train_bpe_tokenizer(input_path, vocab_size, special_tokens, output_dir,
                       dataset_name, time_limit_hours=None, memory_limit_gb=None):
    """
    Train a BPE tokenizer and save results.

    Args:
        input_path: Path to training data
        vocab_size: Vocabulary size
        special_tokens: List of special tokens
        output_dir: Directory to save results
        dataset_name: Name of dataset (for display)
        time_limit_hours: Optional time limit in hours
        memory_limit_gb: Optional memory limit in GB

    Returns:
        tuple: (vocab, merges, elapsed_time, peak_memory, longest_token_decoded)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print("="*70)
    print(f"BPE TOKENIZER TRAINING ON {dataset_name.upper()}")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Input file: {input_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Output directory: {output_dir}")

    # Check if input file exists
    if not input_path.exists():
        print(f"\nError: Input file not found: {input_path}")
        sys.exit(1)

    file_size = input_path.stat().st_size / (1024**3)  # GB
    print(f"  Input file size: {file_size:.2f} GB")

    # Start memory tracking
    tracemalloc.start()

    # Start timing
    start_time = time.time()
    print(f"\nStarting BPE training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Train the tokenizer
        vocab, merges = run_train_bpe(
            input_path=str(input_path),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get memory usage
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Print results
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)

        print(f"\nTime taken: {format_time(elapsed_time)}")
        print(f"Peak memory usage: {format_bytes(peak_memory)}")

        # Analyze the vocabulary
        longest_token_decoded = analyze_vocab(vocab, merges)

        # Save to disk
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70 + "\n")
        save_vocab_and_merges(vocab, merges, output_dir, f"{dataset_name.lower()}_")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n✓ Successfully trained BPE tokenizer on {dataset_name}")
        print(f"✓ Vocabulary size: {len(vocab)}")
        print(f"✓ Number of merges: {len(merges)}")
        print(f"✓ Training time: {format_time(elapsed_time)}")
        print(f"✓ Peak memory: {format_bytes(peak_memory)}")
        print(f"✓ Results saved to: {output_dir}")

        # Check if we meet the requirements
        time_hours = elapsed_time / 3600
        time_minutes = elapsed_time / 60
        memory_gb = peak_memory / (1024**3)

        print(f"\nResource Requirements Check:")
        if time_limit_hours:
            print(f"  Time: {time_hours:.2f} hours ({time_minutes:.2f} min) {'✓' if time_hours <= time_limit_hours else '✗ (exceeds limit)'}")
        else:
            print(f"  Time: {format_time(elapsed_time)}")

        if memory_limit_gb:
            print(f"  Memory: {memory_gb:.2f} GB {'✓' if memory_gb <= memory_limit_gb else '✗ (exceeds limit)'}")
        else:
            print(f"  Memory: {format_bytes(peak_memory)}")

        return vocab, merges, elapsed_time, peak_memory, longest_token_decoded

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
