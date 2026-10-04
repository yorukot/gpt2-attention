"""Compare all heads by layer: uv run python -m src.plot_attention."""

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from transformers import GPT2Tokenizer

from src.data import MODEL_NAME, REVISION

DATA = Path("data/attention")
OUTPUT = Path("figures/head_comparison")
# Passage line and prefix token count. Each prefix is a complete sentence.
PASSAGES = ((485, 14), (670, 24))
LAYERS = range(1, 13)  # Layer numbers start at 1.
ROWS = 6
COLS = 2  # ROWS * COLS must equal the number of heads (12).
FIGSIZE = (12, 30)  # Width and height in inches.
DPI = 100
LINE_COLOR = "#0000FF"
TITLE_FONT = "cmb10"  # Computer Modern Roman Bold, bundled with Matplotlib.


def plot_head(axis, labels: list[str], attention: np.ndarray, title: str) -> None:
    positions = np.arange(len(labels))[::-1]
    queries, keys = np.nonzero(attention)
    lines = [
        ((0, positions[query]), (1, positions[key]))
        for query, key in zip(queries, keys, strict=True)
    ]
    axis.add_collection(
        LineCollection(
            lines,
            colors=LINE_COLOR,
            linewidths=1.5,
            alpha=attention[queries, keys],
        )
    )
    for position, label in zip(positions, labels, strict=True):
        axis.text(-0.04, position, label, ha="right", va="center", fontsize=11)
        axis.text(1.04, position, label, ha="left", va="center", fontsize=11)
    axis.text(-0.04, len(labels), "Query", ha="right", fontsize=10, color="#62656D")
    axis.text(1.04, len(labels), "Attended token", fontsize=10, color="#62656D")
    axis.set(xlim=(-0.65, 1.65), ylim=(-1, len(labels) + 0.8))
    axis.set_title(title, fontsize=14, pad=14, fontfamily=TITLE_FONT)
    axis.set_axis_off()


def main() -> None:
    plt.switch_backend("Agg")
    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_NAME, revision=REVISION, local_files_only=True
    )
    for line, length in PASSAGES:
        sample = torch.load(DATA / f"passage_{line}.pt", weights_only=True)
        token_ids = sample["token_ids"][:length]
        labels = [tokenizer.decode([token]).replace(" ", "·") for token in token_ids]
        # A prefix keeps all past context; its attention needs no renormalization.
        attention = sample["attention"][:, :, :length, :length].numpy()
        directory = OUTPUT / f"passage_{line:04d}"
        directory.mkdir(parents=True, exist_ok=True)

        for layer in LAYERS:
            figure, axes = plt.subplots(
                ROWS, COLS, figsize=FIGSIZE, squeeze=False, layout="constrained"
            )
            for head, (axis, weights) in enumerate(
                zip(axes.flat, attention[layer - 1], strict=True), 1
            ):
                plot_head(axis, labels, weights, f"Head {head}")
            figure.suptitle(f"Layer {layer}", fontsize=18, fontfamily=TITLE_FONT)
            output = directory / f"layer_{layer:02d}.png"
            figure.savefig(output, dpi=DPI, facecolor="white")
            plt.close(figure)
            print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
