import argparse
import math

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer

model_name = "openai-community/gpt2"


@torch.inference_mode()
def gpt2(text: str) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    gpt_2_checkpoint = hf_hub_download(model_name, "model.safetensors")
    weights = load_file(gpt_2_checkpoint, device="cpu")
    return forward(input_ids, weights)[0]


def forward(input_ids: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    batch_size, length = input_ids.shape
    max_length = weights["wpe.weight"].shape[0]
    if not 1 <= length <= max_length:
        raise ValueError(f"Input must contain between 1 and {max_length} tokens.")
    positions = torch.arange(length, device=input_ids.device)

    x = weights["wte.weight"][input_ids] + weights["wpe.weight"][positions]

    mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).tril()

    for layer in range(12):
        prefix = f"h.{layer}"

        normalized = layer_norm(x, weights, prefix + ".ln_1")
        qkv = linear(normalized, weights, prefix + ".attn.c_attn")
        query, key, value = qkv.chunk(3, dim=-1)

        # Split each 768-dimensional embedding into 12 heads of 64 dimensions.
        query = query.reshape(batch_size, length, 12, 64).transpose(1, 2)
        key = key.reshape(batch_size, length, 12, 64).transpose(1, 2)
        value = value.reshape(batch_size, length, 12, 64).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(64)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        context = probabilities @ value

        context = context.transpose(1, 2).reshape(batch_size, length, 768)
        x = x + linear(context, weights, prefix + ".attn.c_proj")

        normalized = layer_norm(x, weights, prefix + ".ln_2")
        hidden = linear(normalized, weights, prefix + ".mlp.c_fc")
        # GPT-2's approximate GELU activation.
        hidden = (
            0.5
            * hidden
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3)))
        )
        x = x + linear(hidden, weights, prefix + ".mlp.c_proj")

    x = layer_norm(x, weights, "ln_f")

    logits = x @ weights["wte.weight"].T
    return logits


def linear(x, weights, name):
    weight = weights[name + ".weight"]
    bias = weights[name + ".bias"]
    output_shape = (*x.shape[:-1], weight.shape[1])
    return torch.addmm(bias, x.reshape(-1, x.shape[-1]), weight).reshape(output_shape)


def layer_norm(x, weights, name):
    return nn.functional.layer_norm(
        x,
        normalized_shape=(768,),
        weight=weights[name + ".weight"],
        bias=weights[name + ".bias"],
        eps=1e-5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GPT-2 forward pass.")
    parser.add_argument(
        "text",
        nargs="?",
        default="The cat sat on the",
        help="Input text (default: %(default)s).",
    )
    args = parser.parse_args()
    if not args.text:
        parser.error("text must not be empty")

    logits = gpt2(args.text)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    next_token_id = logits[-1].argmax().item()
    next_token = tokenizer.decode([next_token_id])

    print(f"Input: {args.text}")
    print(f"Next token: {next_token!r}")


if __name__ == "__main__":
    main()
