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

## Submission

Submit your code and report in the form of a GitHub repository. Send the repository URL to Ak via Slack. The code should reproduce the measurements and figures in the report. The report should include your experimental setup, results, figures, discussion, limitations, references, and a brief statement describing each partner's contributions.

## References

[1] Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher Manning. What Does BERT Look At? An Analysis of BERT's Attention, 2019.
