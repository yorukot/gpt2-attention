import pickletools
import re
from collections.abc import Iterator
from pathlib import Path
from typing import cast

SOURCE = Path("data/raw/bert_attention.pkl")
OUTPUT = Path("data/passages.txt")
OPCODES = {opcode.code: opcode for opcode in pickletools.opcodes}


def read_tokens(path: Path) -> Iterator[list[str]]:
    memo: dict[int, str] = {}
    last_string = None
    expecting_list = False
    tokens = None
    size = path.stat().st_size

    with path.open("rb") as source:
        # check weather the file format is a protocol-2 BERT attention pickle
        if source.read(2) != b"\x80\x02":
            raise ValueError("Expected a protocol-2 BERT attention pickle.")
        while code := source.read(1):
            opcode = OPCODES[code.decode("latin1")]
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
                    memo[cast(int, argument)] = last_string
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


def prepare(source: Path, output: Path) -> int:
    """Write one reconstructed passage per line and return the passage count."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as passages:
        for tokens in read_tokens(source):
            for segment in split_segments(tokens):
                text = detokenize(merge_wordpieces(segment))
                if text:
                    passages.write(text + "\n")
                    count += 1
    return count


# Start the codde
if __name__ == "__main__":
    count = prepare(SOURCE, OUTPUT)
    print(f"Wrote {count} passages to {OUTPUT}.")
