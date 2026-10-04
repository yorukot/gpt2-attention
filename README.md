# Lab 3: Investigating GPT 2 Attention

# Logistics and project brief

- **Format:** Pair project
- **Length:** Two weeks
- **Deliverable:** Approximately 3 to 6 pages, excluding references

In *What Does BERT Look At?* [1], Clark et al. study recurring patterns in BERT's attention heads. Working in pairs, investigate which of these patterns also appear in GPT-2 Small. Your goal is to reproduce selected experiments from the paper, adapt them where GPT-2's causal attention requires a different setup, and explain what you find in a short research-style report.

Use a reproducible collection of text passages and extract the attention weights from your GPT-2 implementation. Your project should include the following analyses.

### Introduction

Write an introduction in your own words that combines the motivation and literature review in the Introduction and Related Work sections of Clark et al. Explain why researchers study attention patterns, summarize the main approaches used to analyze pretrained language models, and introduce the question your report investigates. You do not need to conduct a new literature search, but you are encouraged to read and cite the original papers discussed by Clark et al.

### Surface level patterns

Reproduce and extend the surface-level analyses in Sections 3.1 through 3.3 of Clark et al.

- Measure how much attention each head assigns to the current token, the previous token, and several earlier positions.
- Compare focused and broad attention heads using attention entropy.
- ~~Investigate which heads assign the most attention to GPT-2's `<|endoftext|>` token and compare this behavior with BERT's attention to `[SEP]` and `[CLS]`.~~

GPT-2 cannot attend to future tokens, so the next-token pattern from Section 3.1 is impossible and should simply be discussed as an architectural difference.

~~The `<|endoftext|>` analysis instead adapts the special-token experiment from Section 3.2: investigate whether GPT-2 attends to its document-boundary token in ways resembling BERT's attention to `[SEP]` or `[CLS]`. Use examples in which `<|endoftext|>` is visible to later tokens, such as at the beginning of a passage or between two passages.~~

#### Update October 1, 2026

Since GPT-2's EOS token doesn't appear until the end of text, it is likely OOD if we use it like BERT's `[SEP]` or `[CLS]`. To this end, we redesign this lab to ask these two questions:

- The original paper formulates `[SEP]` as a no-op token. If GPT-2 sequences don't see any tokens that don't have meaning, does the no-op behavior exist in GPT-2? (For example, potential no-op tokens would be stop words such as "the", "a", or punctuation such as "," and ".".)
- **(Optional)** If the no-op behavior doesn't occur in GPT-2, does GPT-2 still have specialized heads? You may choose to skip the investigation of this question as it refers to content in Section 4.

### Clustering attention heads

Redo the experiment in Section 6 of Clark et al. Measure the similarity between attention heads using Jensen-Shannon divergence and visualize the heads in two dimensions using multidimensional scaling. Discuss whether heads from the same layer behave similarly and whether the clusters correspond to the surface-level patterns you found.

### Discussion and conclusion

Bring the experiments together into a broader analysis. Which findings from the BERT paper also appear in GPT-2? Which do not? How much of the difference can be explained by GPT-2's causal architecture? End with the main conclusions that a reader should remember.

### Limitations

Discuss the main limits of your study. These may include the dataset, sequence length, use of only one model size, choices made when averaging attention, and the fact that attention weights alone do not prove that a head causes a model prediction.

## Generating attention data

First extract passages from `data/raw/bert_attention.pkl`:

```bash
uv run python -m src.prepare_text
```

This writes `data/passages.txt`, preserving each BERT segment as one passage.
The source and output paths are constants at the top of `src/prepare_text.py`.

Once `data/passages.txt` exists and the pinned GPT-2 tokenizer and weights are
cached, generate attention data from the repository root:

```bash
uv run python -m src.gpt2_attention
```

`src/data.py` samples 32 passages with seed 42, keeps passages with at least 32
tokens, and caps each at 128 tokens. These settings are constants at the top of
that file. `src/gpt2_attention.py` loads the weights once and runs each passage on
CPU with two threads.

Each passage is saved to `data/attention/passage_<line>.pt`, where `<line>` is its
1-based line number in `passages.txt`. Each file contains the source text, token
IDs, original token count, truncation flag, model name and revision, and an
`attention` tensor shaped `[12, 12, tokens, tokens]`: layer, head, query, key.
The source text is the full passage; the token IDs and attention use its capped
input. Repeating a run overwrites files for the same passage lines.

To read one saved passage:

```python
from pathlib import Path
import torch

path = next(Path("data/attention").glob("*.pt"))
sample = torch.load(path, weights_only=True)
print(sample["token_ids"])
print(sample["attention"].shape)
# First layer, first head; rows are queries and columns are attended tokens.
head_attention = sample["attention"][0, 0]
```

## Plotting attention

After generating the attention files, run:

```bash
uv run python -m src.plot_attention
```

This generates **24 PNGs** at 100 DPI (1200 by 3000 pixels): passages 485 and 670,
each with one image per layer. Every image contains all 12 heads in **6 rows and
2 columns**, ordered left to right and top to bottom. Files are named
`figures/head_comparison/passage_0485/layer_01.png` through `layer_12.png`, with
another 12 files under `passage_0670/`. Repeating a run overwrites the same files.
The title shows only the layer number (for example, `Layer 1`); there is no footer.
Layer and head titles use Computer Modern Roman Bold (`cmb10`, bundled with
Matplotlib).

Change the constants at the top of `src/plot_attention.py`; there are no command
line options:

- `PASSAGES`: pairs of passage line number and prefix token count.
- `LAYERS`: layer numbers to plot, starting at 1.
- `ROWS`, `COLS`: grid layout; their product must equal 12. For example, set
  `ROWS = 3` and `COLS = 4` for three rows of four heads.
- `FIGSIZE`: figure width and height in inches; adjust alongside the layout
  (for example, `(24, 18)` for a 3-by-4 grid).
- `DPI`: output resolution; 100 for previews, or 200 for larger images.
- `LINE_COLOR`: attention line color; defaults to pure blue (`#0000FF`).
- `TITLE_FONT`: font family for layer and head titles.
- `OUTPUT`: output directory.

Each panel shows query tokens on the left and attended tokens on the right.
All nonzero weights are drawn, with line opacity equal to the weight. Labels
retain GPT-2's token splits, with `·` indicating a space.

Passage lines refer to `data/passages.txt`. The default prefixes end at complete
sentences and keep all their original past context. The script uses the saved
attention directly, without rerunning GPT-2 or renormalizing the weights.

## Submission

Submit your code and report in the form of a GitHub repository. Send the repository URL to Ak via Slack. The code should reproduce the measurements and figures in the report. The report should include your experimental setup, results, figures, discussion, limitations, references, and a brief statement describing each partner's contributions.

## References

[1] Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher Manning. What Does BERT Look At? An Analysis of BERT's Attention, 2019.
