import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

from src.data import MODEL_NAME, REVISION, load_data

OUTPUT = Path("data/attention")
THREADS = 2


def forward(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    attention_maps: list[torch.Tensor] | None = None,
) -> torch.Tensor:
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

        query = query.reshape(batch_size, length, 12, 64).transpose(1, 2)
        key = key.reshape(batch_size, length, 12, 64).transpose(1, 2)
        value = value.reshape(batch_size, length, 12, 64).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(64)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        if attention_maps is not None:
            attention_maps.append(probabilities)
        context = probabilities @ value

        context = context.transpose(1, 2).reshape(batch_size, length, 768)
        x = x + linear(context, weights, prefix + ".attn.c_proj")

        normalized = layer_norm(x, weights, prefix + ".ln_2")
        hidden = linear(normalized, weights, prefix + ".mlp.c_fc")
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


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(THREADS)
    _, passages = load_data()
    checkpoint = hf_hub_download(
        MODEL_NAME, "model.safetensors", revision=REVISION, local_files_only=True
    )
    weights = load_file(checkpoint, device="cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for index, passage in enumerate(passages, 1):
        input_ids = torch.tensor([passage["token_ids"]], dtype=torch.long)
        maps = []
        forward(input_ids, weights, attention_maps=maps)
        # Remove the single-passage batch dimension: [layer, head, query, key].
        attention = torch.stack(maps)[:, 0]
        output = OUTPUT / f"passage_{passage['passage_line']}.pt"
        torch.save(
            {
                **passage,
                "model": MODEL_NAME,
                "revision": REVISION,
                "attention": attention,
            },
            output,
        )
        print(f"[{index}/{len(passages)}] Saved {output}", flush=True)


if __name__ == "__main__":
    main()
