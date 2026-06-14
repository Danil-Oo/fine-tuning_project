"""
Модуль вычисления perplexity на val-сете.
"""

import json

import torch
from tqdm import tqdm


def load_val_data(path: str, n: int = None) -> list[dict]:
    """Загружает данные из JSONL файла."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if n is not None:
        items = items[:n]
    return items


def compute_perplexity(
    model,
    tokenizer,
    val_data: list[dict],
    device: str,
    max_length: int = 512,
) -> dict:
    """
    Вычисляет perplexity на val-сете.

    Возвращает словарь:
      {
        "perplexity": float,
        "avg_loss": float,
        "n_samples": int,
        "perplexity_by_category": dict  # если есть поле 'category'
      }
    """
    model.eval()

    total_loss = 0.0
    count = 0
    per_category: dict[str, list[float]] = {}

    with torch.no_grad():
        for item in tqdm(val_data, desc="Computing perplexity"):
            msgs = item.get("messages", [])
            if not msgs:
                continue

            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

            total_loss += loss
            count += 1

            category = item.get("category", "unknown")
            per_category.setdefault(category, []).append(loss)

    if count == 0:
        raise ValueError("No valid samples found in val data")

    avg_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    perplexity_by_category = {
        cat: torch.exp(torch.tensor(sum(losses) / len(losses))).item()
        for cat, losses in per_category.items()
    }

    return {
        "perplexity": round(perplexity, 4),
        "avg_loss": round(avg_loss, 6),
        "n_samples": count,
        "perplexity_by_category": perplexity_by_category,
    }
