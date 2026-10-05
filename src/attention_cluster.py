"""Cluster heads by Jensen-Shannon distance: uv run python -m src.cluster."""

import math
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

DATA = Path("data/attention")
DISTANCES = Path("data/head_distances.pt")
OUTPUT = Path("figures/head_clusters")
COMMA = 11
PERIOD = 13
ENRICHMENT = 3
BROAD = 0.8
BEHAVIORS = {
    "Current token": "#2A9D8F",
    "Previous token": "#E76F51",
    '","': "#7B2CBF",
    '"."': "#E9C46A",
    "Induction": "#3A86FF",
    "Broad": "#8AB17D",
    "First token": "#9AA5B8",
}
SPECIFIC = ("Current token", "Previous token", '","', '"."', "Induction")
OTHER_COLOR = "#E3E5EA"
SEED = 42
FIGSIZE = (10, 10)  # Width and height in inches.
DPI = 100
COLORMAP = "viridis"
TITLE_FONT = "cmb10"  # Computer Modern Roman Bold, bundled with Matplotlib.


def entropy(attention: torch.Tensor) -> torch.Tensor:
    """Entropy in bits of each row; xlogy treats 0 * log 0 as 0."""
    return -torch.special.xlogy(attention, attention).sum(-1) / math.log(2)


def passage_divergences(attention: torch.Tensor) -> torch.Tensor:
    heads = attention.shape[0]
    row_entropy = entropy(attention)  # [heads, query]
    total = torch.zeros(heads, heads, dtype=attention.dtype)
    # One head against all others at a time keeps memory at [heads, query, key].
    for head in range(heads):
        mixture = entropy((attention[head] + attention) / 2)
        divergence = mixture - (row_entropy[head] + row_entropy) / 2
        total[head] = divergence.sum(-1)
    return total


def head_distances() -> torch.Tensor:
    total = torch.zeros(144, 144, dtype=torch.float64)
    tokens = 0
    paths = sorted(DATA.glob("*.pt"))
    for index, path in enumerate(paths, 1):
        attention = torch.load(path, weights_only=True)["attention"].double()
        length = attention.shape[-1]
        total += passage_divergences(attention.reshape(144, length, length))
        tokens += length
        print(f"[{index}/{len(paths)}] {path.name}: {length} tokens", flush=True)
    distances = total / tokens
    # Clamp rounding noise so MDS gets an exact, symmetric dissimilarity matrix.
    distances = ((distances + distances.T) / 2).clamp_min(0)
    distances.fill_diagonal_(0)
    return distances


def target_masks(token_ids: list[int]) -> dict[str, torch.Tensor]:
    """Query-key cells, [query, key], that count as each behavior's target."""
    length = len(token_ids)
    tokens = torch.tensor(token_ids)
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    current = torch.eye(length, dtype=torch.bool)
    # Query 0 can only attend to itself, which says nothing about the head.
    current[0, 0] = False
    return {
        "Current token": current,
        "Previous token": torch.ones(length, length, dtype=torch.bool)
        .diag(-1)
        .diag(-1),
        '","': causal & (tokens == COMMA),
        '"."': causal & (tokens == PERIOD),
        # The token right after an earlier copy of the query token.
        "Induction": causal.tril(-1)
        & torch.cat(
            [
                torch.zeros(length, 1, dtype=torch.bool),
                tokens[:, None] == tokens[None, :-1],
            ],
            1,
        ),
        # Query 0 is left out for the same reason as above.
        "First token": torch.zeros(length, length, dtype=torch.bool).index_fill(
            1, torch.tensor(0), True
        )
        & ~torch.eye(length, dtype=torch.bool),
    }


def head_behaviors() -> list[str | None]:
    """Behavior of each head in BEHAVIORS order, or None when it has none."""
    targets = (*SPECIFIC, "First token")
    observed = {name: torch.zeros(144, dtype=torch.float64) for name in targets}
    expected = dict.fromkeys(targets, 0.0)
    head_entropy = torch.zeros(144, dtype=torch.float64)
    uniform_entropy = 0.0
    for path in sorted(DATA.glob("*.pt")):
        sample = torch.load(path, weights_only=True)
        length = len(sample["token_ids"])
        attention = sample["attention"].double().reshape(144, length, length)
        # A uniform causal head spreads each query evenly over the tokens it sees.
        uniform = torch.ones(length, length, dtype=torch.float64).tril()
        uniform /= uniform.sum(-1, keepdim=True)
        for name, mask in target_masks(sample["token_ids"]).items():
            observed[name] += attention[:, mask].sum(-1)
            expected[name] += uniform[mask].sum().item()
        # Query 0 always has zero entropy, so only later queries count.
        head_entropy += entropy(attention[:, 1:]).sum(-1)
        uniform_entropy += entropy(uniform[1:]).sum().item()

    ratios = {name: observed[name] / expected[name] for name in targets}
    specific = torch.stack([ratios[name] for name in SPECIFIC]).max(0)
    broad = head_entropy / uniform_entropy
    behaviors = []
    for head in range(144):
        if specific.values[head] >= ENRICHMENT:
            behaviors.append(SPECIFIC[specific.indices[head]])
        elif broad[head] >= BROAD:
            behaviors.append("Broad")
        elif ratios["First token"][head] >= ENRICHMENT:
            behaviors.append("First token")
        else:
            behaviors.append(None)
    return behaviors


def scatter_heads(points, colors) -> tuple[plt.Figure, plt.Axes, object]:
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    scatter = axis.scatter(points[:, 0], points[:, 1], c=colors, s=60, alpha=0.85)
    for index, (x, y) in enumerate(points):
        axis.annotate(
            f"{index // 12 + 1}-{index % 12 + 1}",
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color="#62656D",
        )
    axis.set_title(
        "Attention heads by Jensen-Shannon distance (MDS)",
        fontsize=16,
        fontfamily=TITLE_FONT,
    )
    # MDS axes have no intrinsic meaning; only distances between points matter.
    axis.set_xticks([])
    axis.set_yticks([])
    figure.supxlabel(
        "Point label: layer-head. Closer points have more similar attention "
        "(lower mean JS divergence per token).",
        fontsize=9,
        color="#62656D",
    )
    return figure, axis, scatter


def save(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output, dpi=DPI, facecolor="white")
    plt.close(figure)
    print(f"Saved {output}", flush=True)


def plot_layers(points) -> None:
    figure, axis, scatter = scatter_heads(points, torch.arange(144) // 12 + 1)
    scatter.set_cmap(COLORMAP)
    figure.colorbar(scatter, ax=axis, label="Layer", ticks=range(1, 13))
    save(figure, OUTPUT / "mds.png")


def plot_behaviors(points, behaviors: list[str | None]) -> None:
    colors = [BEHAVIORS.get(behavior, OTHER_COLOR) for behavior in behaviors]
    figure, axis, _ = scatter_heads(points, colors)
    labels = {**BEHAVIORS, "Other": OTHER_COLOR}
    counts = {name: behaviors.count(name) for name in BEHAVIORS}
    counts["Other"] = behaviors.count(None)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=8)
        for color in labels.values()
    ]
    legend = axis.legend(
        handles,
        [f"{name} ({counts[name]})" for name in labels],
        title=f"Behavior ({ENRICHMENT}x uniform; broad: {BROAD:.0%} entropy)",
        loc="lower right",
    )
    # Lower the bottom of the plot until no point or label sits under the legend.
    renderer = figure.canvas.get_renderer()
    bottom, top = axis.get_ylim()
    while True:
        figure.canvas.draw()
        # Padding covers the marker radius around each point center.
        box = legend.get_window_extent(renderer).padded(10)
        centers = axis.transData.transform(points)
        covered = any(box.contains(x, y) for x, y in centers) or any(
            box.overlaps(text.get_window_extent(renderer)) for text in axis.texts
        )
        if not covered:
            break
        bottom -= (top - bottom) * 0.02
        axis.set_ylim(bottom, top)
    save(figure, OUTPUT / "mds_behavior.png")


@torch.inference_mode()
def main() -> None:
    plt.switch_backend("Agg")
    distances = head_distances()
    torch.save({"distances": distances}, DISTANCES)
    print(f"Saved {DISTANCES}", flush=True)

    # Every figure shares one embedding so the heads stay in the same places.
    mds = MDS(metric="precomputed", init="random", n_init=4, random_state=SEED)
    points = mds.fit_transform(distances.numpy())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_layers(points)
    plot_behaviors(points, head_behaviors())


if __name__ == "__main__":
    main()
