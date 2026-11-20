"""Run the pinned, pre-2024 ManifestoBERTa sentence classifier on individual statements."""

import numpy as np
import torch
from model_files import model_dir
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification


def classify_topics(rows):
    path = model_dir("topics")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(5)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    tok.enable_truncation(max_length=256)
    windows = []
    for i, r in enumerate(rows):
        enc = tok.encode(r["text"])
        windows.extend((i, w.ids) for w in [enc, *enc.overflowing])
    windows.sort(key=lambda e: len(e[1]))
    probs = np.zeros((len(rows), model.config.num_labels))
    total = np.zeros(len(rows))
    batch_size = 24 if device == "cuda" else 8
    with torch.inference_mode():
        for n in range(0, len(windows), batch_size):
            batch = windows[n : n + batch_size]
            ids = np.full(
                (len(batch), len(batch[-1][1])),
                model.config.pad_token_id,
                dtype=np.int64,
            )
            mask = np.zeros_like(ids)
            for j, (_, seq) in enumerate(batch):
                ids[j, : len(seq)] = seq
                mask[j, : len(seq)] = 1
            logits = model(
                input_ids=torch.from_numpy(ids).to(device),
                attention_mask=torch.from_numpy(mask).to(device),
            ).logits
            for (i, seq), p in zip(batch, logits.float().softmax(-1).cpu().numpy()):
                weight = max(1, len(seq) - 2)
                probs[i] += p * weight
                total[i] += weight
            if n % 2400 == 0:
                print("topic windows", n, len(windows), device, flush=True)
    return [model.config.id2label[int(p.argmax())] for p in probs / total[:, None]]
