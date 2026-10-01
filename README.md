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
- Investigate which heads assign the most attention to GPT-2's `<|endoftext|>` token and compare this behavior with BERT's attention to `[SEP]` and `[CLS]`.

These are two separate comparisons. GPT-2 cannot attend to future tokens, so the next-token pattern from Section 3.1 is impossible and should simply be discussed as an architectural difference. The `<|endoftext|>` analysis instead adapts the special-token experiment from Section 3.2: investigate whether GPT-2 attends to its document-boundary token in ways resembling BERT's attention to `[SEP]` or `[CLS]`. Use examples in which `<|endoftext|>` is visible to later tokens, such as at the beginning of a passage or between two passages.

### Clustering attention heads

Redo the experiment in Section 6 of Clark et al. Measure the similarity between attention heads using Jensen-Shannon divergence and visualize the heads in two dimensions using multidimensional scaling. Discuss whether heads from the same layer behave similarly and whether the clusters correspond to the surface-level patterns you found.

### Discussion and conclusion

Bring the experiments together into a broader analysis. Which findings from the BERT paper also appear in GPT-2? Which do not? How much of the difference can be explained by GPT-2's causal architecture? End with the main conclusions that a reader should remember.

### Limitations

Discuss the main limits of your study. These may include the dataset, sequence length, use of only one model size, choices made when averaging attention, and the fact that attention weights alone do not prove that a head causes a model prediction.

## Submission

Submit your code and report in the form of a GitHub repository. Send the repository URL to Ak via Slack. The code should reproduce the measurements and figures in the report. The report should include your experimental setup, results, figures, discussion, limitations, references, and a brief statement describing each partner's contributions.

## References

[1] Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher Manning. What Does BERT Look At? An Analysis of BERT's Attention, 2019.

## Python environment

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/). Install the
locked dependencies into the local `.venv`:

```bash
uv sync --locked
```

Run scripts and notebooks in that environment:

```bash
uv run python your_script.py
uv run jupyter lab
```

The environment includes:

- **Models and data:** PyTorch, Transformers, Datasets, Hugging Face Hub,
  Tokenizers, Safetensors, and Accelerate.
- **Attention analysis:** NumPy, SciPy (entropy and Jensen-Shannon distance),
  scikit-learn (multidimensional scaling and clustering), pandas, and einops.
- **Figures and progress:** Matplotlib, Seaborn, and tqdm.
- **Development and notebooks:** JupyterLab, ipykernel, ipywidgets, pytest, and
  Ruff. These are in the `dev` dependency group, installed by default.

PyTorch is configured to use CPU wheels on Linux and Windows, with PyPI wheels
on macOS. For an NVIDIA GPU, update the PyTorch index in `pyproject.toml` to
match the driver and accelerator, then run `uv lock` and `uv sync`. See
[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

When extracting attention weights with Transformers, load GPT-2 with
`attn_implementation="eager"`, use `model.eval()`, and pass
`output_attentions=True` during inference. This makes the attention matrices
available for the analyses in this brief. See the
[Hugging Face GPT-2 documentation](https://huggingface.co/docs/transformers/en/model_doc/gpt2).

Keep `uv.lock` in Git so both partners use the same dependency versions. The
`.gitignore` excludes virtual environments, caches, local `.env` files,
downloaded data/model directories, and experiment outputs. Save figures needed
for the report outside the ignored directories, for example in `report/figures/`.
