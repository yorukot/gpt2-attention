"""Extract readable passages and conservative sentence candidates from BERT data.

Usage: python src/prepare_text.py data/raw/bert_attention.pkl

The source is a protocol-2 pickle of token lists and NumPy attention arrays.
Read its opcodes without unpickling objects, and seek past the array payloads.
Sentence splitting is heuristic: the source has already lost casing, spacing,
and some sentence boundaries. All segments are retained in passages.txt.
"""

import argparse
import hashlib
import json
import pickletools
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "rev",
    "st",
    "sr",
    "jr",
    "vs",
    "etc",
    "fig",
    "no",
    "vol",
    "pp",
    "ed",
    "eds",
    "inc",
    "ltd",
    "co",
    "approx",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "dept",
    "gen",
    "col",
    "lt",
    "capt",
    "sgt",
    "mt",
}


def read_tokens(path: Path) -> Iterator[list[str]]:
    """Read this dataset's token lists without executing pickle instructions."""
    memo: dict[int, str] = {}
    last_string = None
    expecting_list = False
    tokens = None
    size = path.stat().st_size

    with path.open("rb") as source:
        if source.read(2) != b"\x80\x02":
            raise ValueError("Expected a protocol-2 BERT attention pickle.")
        while code := source.read(1):
            opcode = pickletools.code2op[code.decode("latin1")]
            name = opcode.name
            if name == "BINSTRING":
                # NumPy's raw attention data: skip gigabytes of unused values.
                length_bytes = source.read(4)
                if len(length_bytes) != 4:
                    raise ValueError("Truncated binary length.")
                length = int.from_bytes(length_bytes, "little", signed=True)
                if length < 0 or source.tell() + length > size:
                    raise ValueError("Invalid binary payload length.")
                source.seek(length, 1)
                argument = None
            else:
                argument = opcode.arg.reader(source) if opcode.arg else None

            if name in {"BINPUT", "LONG_BINPUT"}:
                if last_string is not None:
                    memo[argument] = last_string
                else:
                    memo.pop(argument, None)
                continue

            if name in {"BINGET", "LONG_BINGET"}:
                value = memo.get(argument)
                if tokens is not None and value is None:
                    raise ValueError("Unresolved string reference in token list.")
            elif name in {"BINUNICODE", "SHORT_BINSTRING"}:
                value = argument
            else:
                value = None

            if expecting_list and name == "EMPTY_LIST":
                tokens = []
                expecting_list = False
            elif tokens is not None:
                if value is not None:
                    tokens.append(value)
                elif name == "APPENDS":
                    if not tokens or tokens[0] != "[CLS]" or tokens[-1] != "[SEP]":
                        raise ValueError("Unexpected BERT token-list structure.")
                    yield tokens
                    tokens = None
                elif name != "MARK":
                    raise ValueError(f"Unexpected {name} inside a token list.")
            elif value == "tokens":
                expecting_list = True

            last_string = value
            if name == "STOP":
                if source.tell() != size or tokens is not None or expecting_list:
                    raise ValueError("Incomplete record or trailing pickle data.")
                return
        raise ValueError("Truncated pickle: missing STOP instruction.")


def split_segments(tokens: list[str]) -> Iterator[list[str]]:
    segment = []
    for token in tokens:
        if token == "[SEP]":
            if segment:
                yield segment
                segment = []
        elif token not in {"[CLS]", "[PAD]"}:
            segment.append(token)
    if segment:
        yield segment


def merge_wordpieces(tokens: list[str]) -> list[str]:
    words: list[str] = []
    for token in tokens:
        if token.startswith("##") and words:
            words[-1] += token[2:]
        else:
            words.append(token.removeprefix("##"))
    return words


def detokenize(words: list[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"\b(\w+)\s+(['’])\s+(s|t|re|ve|ll|d|m)\b", r"\1\2\3", text)
    text = re.sub(r"\b(\w+s)\s+(['’])(?=\s|$)", r"\1\2", text)
    text = re.sub(r"\b([odl])\s+(['’])\s+(\w+)\b", r"\1\2\3", text)
    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)
    text = re.sub(r"(?<!\w)'\s+(.*?)\s+'(?!\w)", r"'\1'", text)
    text = re.sub(r"(?<=\d)\s*([.,:/])\s*(?=\d)", r"\1", text)
    text = re.sub(
        r"\b(?:[a-zA-Z]\s+\.\s*){2,}",
        lambda match: re.sub(r"\s+", "", match[0]) + " ",
        text,
    )
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"(?<=\d)\s*([–—])\s*(?=\d)", r"\1", text)
    return " ".join(text.split())


def sentence_chunks(words: list[str]) -> Iterator[tuple[list[str], bool]]:
    """Yield punctuation-delimited chunks and whether each has an ending."""
    start = 0
    index = 0
    while index < len(words):
        token = words[index]
        previous = words[index - 1] if index else ""
        following = words[index + 1] if index + 1 < len(words) else ""
        boundary = token in {".", "!", "?"}
        if token == ".":
            boundary = not (
                (previous.isdigit() and following.isdigit())
                or previous.lower() in ABBREVIATIONS
                or (len(previous) == 1 and previous.isalpha())
                or previous == "."
                or following == "."
            )
        if boundary:
            end = index + 1
            while end < len(words):
                if (
                    words[end] in {"!", "?", ")", "]", "}"}
                    or words[end] == '"'
                    and words[start:end].count('"') % 2
                ):
                    end += 1
                else:
                    break
            yield words[start:end], True
            start = end
            index = end
        else:
            index += 1
    if start < len(words):
        yield words[start:], False


def exclusion_reason(words: list[str], index: int, has_ending: bool) -> str | None:
    if index == 0:
        return "unverified_segment_start"
    if not has_ending:
        return "unfinished_segment_end"
    if "[UNK]" in words:
        return "unknown_token"
    if sum(any(char.isalpha() for char in word) for word in words) < 3:
        return "too_short"
    if words[0] in {",", ";", ":", ")", "]", "}"}:
        return "leading_fragment"
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if words.count(opening) != words.count(closing):
            return "unbalanced_brackets"
    if words.count('"') % 2:
        return "unbalanced_quotes"
    return None


def prepare(source: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    exclusions: Counter = Counter()
    seen: set[str] = set()
    with (
        (output_dir / "passages.txt").open("w", encoding="utf-8") as passages,
        (output_dir / "sentences.txt").open("w", encoding="utf-8") as sentences,
        (output_dir / "segments.jsonl").open("w", encoding="utf-8") as records,
    ):
        for record_id, tokens in enumerate(read_tokens(source)):
            counts["source_records"] += 1
            counts["source_tokens_including_special_tokens"] += len(tokens)
            for segment_id, segment in enumerate(split_segments(tokens)):
                words = merge_wordpieces(segment)
                text = detokenize(words)
                if not text:
                    continue
                counts["passages"] += 1
                passages.write(text + "\n")
                chunks = []
                for index, (chunk, has_ending) in enumerate(sentence_chunks(words)):
                    sentence = detokenize(chunk)
                    reason = exclusion_reason(chunk, index, has_ending)
                    if reason is None and sentence in seen:
                        reason = "duplicate_sentence"
                    item = {"text": sentence, "excluded_reason": reason}
                    if reason is None:
                        seen.add(sentence)
                        counts["sentences"] += 1
                        item["sentence_line"] = counts["sentences"]
                        sentences.write(sentence + "\n")
                    else:
                        exclusions[reason] += 1
                    chunks.append(item)
                record = {
                    "record_id": record_id,
                    "segment_id": segment_id,
                    "passage_line": counts["passages"],
                    "source_tokens": segment,
                    "text": text,
                    "chunks": chunks,
                }
                records.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not counts["source_records"]:
        raise ValueError("No BERT records found.")
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    summary = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": digest,
        "counts": dict(counts),
        "excluded_chunks": dict(exclusions),
        "notes": [
            "UTF-8 outputs use one passage or sentence candidate per line.",
            "Passages preserve each BERT segment separately; no segments are joined.",
            "Stop words and punctuation are retained; no EOS tokens are inserted.",
            "WordPiece continuations are joined and punctuation spacing is repaired.",
            "Original casing, spacing, and clipped text cannot be recovered exactly.",
            "Sentence splitting is heuristic and does not guarantee grammatical completeness.",
            "The first chunk of every segment is excluded because its start is uncertain.",
            "Unfinished endings and duplicate sentence candidates are excluded.",
            "All excluded text remains in passages.txt and segments.jsonl.",
            "Line numbers are 1-based; record and segment IDs are 0-based.",
        ],
    }
    (output_dir / "preprocessing.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    summary = prepare(args.source, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
