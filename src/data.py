import random
from pathlib import Path

from transformers import GPT2Tokenizer

MODEL_NAME = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATASET = Path("data/passages.txt")
PASSAGES = 32
MIN_TOKENS = 32
MAX_TOKENS = 128
SEED = 42


def load_data():
    """Return the tokenizer and sampled passages, including their token IDs."""
    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_NAME, revision=REVISION, local_files_only=True
    )
    eligible = []
    for line_number, text in enumerate(
        DATASET.read_text(encoding="utf-8").splitlines(), 1
    ):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) >= MIN_TOKENS:
            eligible.append(
                {
                    "passage_line": line_number,
                    "source_text": text,
                    "original_token_count": len(token_ids),
                    "truncated": len(token_ids) > MAX_TOKENS,
                    "token_ids": token_ids[:MAX_TOKENS],
                }
            )
    return tokenizer, random.Random(SEED).sample(eligible, PASSAGES)
