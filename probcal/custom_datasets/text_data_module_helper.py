import os
from typing import Iterable
import torch
from transformers import BatchEncoding
from transformers import DistilBertTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

def collate_fn(
    batch: Iterable[tuple[str, int]],
) -> tuple[BatchEncoding, torch.LongTensor]:
    all_text = []
    all_targets = []
    for x in batch:
        all_text.append(x[0])
        all_targets.append(x[1])

    input_tokens = tokenizer(
        all_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    targets_tensor = torch.tensor(all_targets)

    return input_tokens, targets_tensor