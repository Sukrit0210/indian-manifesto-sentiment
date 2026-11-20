"""Run a frozen PyTorch transformer. Long inputs use all 256-token windows."""

import numpy as np
import torch
from model_files import model_dir
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def infer_sentiment(texts):
    folder = model_dir("roberta")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True).to(device).eval()
    if [model.config.id2label[i].lower() for i in range(3)] != [
        "negative",
        "neutral",
        "positive",
    ]:
        raise ValueError("Unexpected sentiment label order")
    windows = []
    for row, text in enumerate(texts):
        encoded = tokenizer(
            text, truncation=True, max_length=256, return_overflowing_tokens=True
        )
        windows.extend((row, ids) for ids in encoded["input_ids"])
    windows.sort(key=lambda item: len(item[1]))
    width = model.config.num_labels
    sums = np.zeros((len(texts), width), dtype=np.float64)
    counts = np.zeros((len(texts), 1), dtype=np.float64)
    batch_size = 32 if device.type == "cuda" else 8
    print(f"Sentiment: {len(texts)} passages on {device}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            inputs = tokenizer.pad(
                [{"input_ids": ids} for _, ids in batch], return_tensors="pt"
            ).to(device)
            values = model(**inputs).logits.softmax(dim=1)
            for (row, ids), value in zip(batch, values.cpu().numpy()):
                weight = max(1, len(ids) - 2)
                sums[row] += value * weight
                counts[row] += weight
    return (sums / counts).astype(np.float32)
