#!/usr/bin/env python3
"""
Fix vocab.json for already-trained tokenizers.
Regenerate vocab.json from vocab_raw.json using the corrected format.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.common import gpt2_bytes_to_unicode


def fix_vocab_json(tokenizer_dir, name_prefix):
    """Fix vocab.json from vocab_raw.json."""
    tokenizer_dir = Path(tokenizer_dir)

    print(f"Fixing vocab.json in: {tokenizer_dir}")

    # Load raw vocab
    raw_vocab_path = tokenizer_dir / f"{name_prefix}_vocab_raw.json"
    with open(raw_vocab_path) as f:
        vocab_raw = json.load(f)

    print(f"  Loaded {len(vocab_raw)} tokens from {raw_vocab_path.name}")

    # Convert to GPT-2 format (token_id -> token_string)
    byte_encoder = gpt2_bytes_to_unicode()
    vocab_gpt2 = {}

    for token_id_str, hex_bytes in vocab_raw.items():
        token_bytes = bytes.fromhex(hex_bytes)
        token_str = ''.join(byte_encoder[b] for b in token_bytes)
        vocab_gpt2[token_id_str] = token_str  # Use token_id as key

    # Save corrected vocab.json
    vocab_path = tokenizer_dir / f"{name_prefix}_vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_gpt2, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved {len(vocab_gpt2)} tokens to {vocab_path.name}")

    # Verify
    min_id = min(int(k) for k in vocab_gpt2.keys())
    max_id = max(int(k) for k in vocab_gpt2.keys())
    print(f"  ✓ Token IDs: {min_id} to {max_id}")
    print(f"  ✓ Fixed successfully!\n")


if __name__ == "__main__":
    print("="*70)
    print("FIX VOCAB.JSON FOR TRAINED TOKENIZERS")
    print("="*70 + "\n")

    # Fix TinyStories
    tinystories_dir = Path("/home/lifans/cs336/cs336-assignment1-basics/tokenizer_output/tinystories")
    if tinystories_dir.exists():
        fix_vocab_json(tinystories_dir, "tinystories")
    else:
        print(f"TinyStories tokenizer not found at {tinystories_dir}\n")

    # Fix OpenWebText (if exists)
    owt_dir = Path("/home/lifans/cs336/cs336-assignment1-basics/tokenizer_output/openwebtext")
    if owt_dir.exists():
        fix_vocab_json(owt_dir, "openwebtext")
    else:
        print(f"OpenWebText tokenizer not found at {owt_dir}\n")

    print("="*70)
    print("ALL DONE!")
    print("="*70)
