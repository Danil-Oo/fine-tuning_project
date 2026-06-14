"""
02_prepare_dataset.py

Что делает этот скрипт:
1. Загружает Q&A пары из базы знаний (data/generated/qa_pairs_ru.json)
2. Удаляет дубликаты по похожести вопросов (SequenceMatcher)
3. Загружает coding датасет с HuggingFace
4. Делает стратифицированную выборку coding примеров по бакетам длин
5. Конвертирует всё в chat-формат Qwen
6. Перемешивает и делает train/val сплит 90/10
7. Сохраняет в data/final/train.jsonl и data/final/val.jsonl
"""

import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Системные промпты
# ---------------------------------------------------------------------------

SYSTEM_RU = (
    "Ты — технический ассистент, специализирующийся на архитектуре больших "
    "языковых моделей. Отвечай точно и развёрнуто на русском языке."
)
SYSTEM_EN = (
    "You are a helpful coding assistant. "
    "Provide clear, working Python code with brief explanations."
)


# ---------------------------------------------------------------------------
# Конвертация в chat-формат
# ---------------------------------------------------------------------------


def qa_to_chat(item: dict) -> dict:
    """
    Конвертирует одну Q&A пару в chat-формат Qwen.

    Вход:  {"question": "...", "answer": "...", "source_file": "..."}
    Выход: {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

    Метаданные (source_file, chunk_index и т.д.) игнорируются — они больше не нужны.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RU},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
    }


def coding_to_chat(item: dict) -> dict:
    """
    Конвертирует один пример из Alpaca-датасета в chat-формат Qwen.

    Alpaca-формат: {"instruction": "...", "input": "...", "output": "..."}

    Поле "input" часто пустое. Если оно есть — добавляем к instruction.
    Используем чистые поля (не "prompt") чтобы избежать Alpaca-шаблона
    вида "### Instruction: ... ### Response:".
    """
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["output"]},
        ]
    }


# ---------------------------------------------------------------------------
# Дедупликация Q&A пар
# ---------------------------------------------------------------------------


def deduplicate_qa(pairs: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Удаляет Q&A пары с очень похожими вопросами.

    SequenceMatcher(None, str1, str2).ratio() возвращает число 0.0–1.0:
      1.0 = строки идентичны
      0.85+ = строки очень похожи → считаем дубликатом

    Алгоритм: для каждого нового вопроса сравниваем со всеми уже
    принятыми. Если хотя бы одно совпадение выше порога — пропускаем.

    threshold=0.85 ловит вопросы с разным порядком слов, с/без скобок,
    с небольшими перефразировками.
    """
    unique = []
    accepted_questions = []  # только тексты вопросов для сравнения

    for pair in pairs:
        q = pair["question"].lower()

        is_duplicate = any(
            SequenceMatcher(None, q, existing).ratio() > threshold
            for existing in accepted_questions
        )

        if not is_duplicate:
            unique.append(pair)
            accepted_questions.append(q)

    print(
        f"  Дедупликация: {len(pairs)} → {len(unique)} пар "
        f"(удалено {len(pairs) - len(unique)})"
    )
    return unique


# ---------------------------------------------------------------------------
# Загрузка и стратифицированная выборка coding датасета
# ---------------------------------------------------------------------------


def load_coding_dataset() -> pd.DataFrame:
    """
    Загружает датасет с HuggingFace и добавляет колонку output_len.
    """
    print("  Загружаю coding датасет с HuggingFace...")
    dataset = load_dataset(
        "iamtarun/python_code_instructions_18k_alpaca", split="train"
    )
    df = dataset.to_pandas()
    df["output_len"] = df["output"].str.len()
    print(f"  Загружено примеров: {len(df)}")
    return df


def sample_coding_balanced(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Стратифицированная выборка по бакетам длины output.

    Бакеты и целевые количества:
      260–500  символов → 70  (много простых примеров, берём немного)
      500–800  символов → 160 (основа, лучшее качество)
      800–1100 символов → 90  (более сложные задачи)
      1100–1386 символов → 50 (длинные, берём избирательно)
    Итого: 370 примеров

    Если в бакете меньше примеров чем нужно — берём все доступные.
    Верхняя граница 1386 выровнена по максимальной длине Q&A пар.
    """
    buckets = [
        (260, 500, 70),
        (500, 800, 160),
        (800, 1100, 90),
        (1100, 1386, 50),
    ]

    sampled_parts = []
    for lo, hi, target in buckets:
        bucket = df[(df["output_len"] >= lo) & (df["output_len"] < hi)]
        n = min(target, len(bucket))
        sampled_parts.append(bucket.sample(n=n, random_state=seed))
        print(f"  Бакет {lo:>4}–{hi:<4}: доступно {len(bucket):>5}, взято {n}")

    result = pd.concat(sampled_parts).reset_index(drop=True)
    print(f"  Итого coding примеров: {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Сохранение в JSONL
# ---------------------------------------------------------------------------


def save_jsonl(data: list[dict], path: str) -> None:
    """
    Сохраняет список словарей в JSONL-файл (одна строка = один JSON-объект).

    JSONL используется вместо JSON потому что при обучении файл читается
    построчно — это быстрее и не требует загружать весь файл в память.

    ensure_ascii=False — кириллица сохраняется как есть, не как \\uXXXX.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------


def build_dataset():
    print("=" * 55)
    print("Шаг 1. Загрузка и дедупликация Q&A пар")
    print("=" * 55)

    with open("data/generated/qa_pairs_ru.json", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"  Загружено Q&A пар: {len(qa_pairs)}")

    # Дедупликация происходит ДО конвертации в chat-формат —
    # на сырых {"question": ..., "answer": ...} сравнивать проще и быстрее
    qa_pairs = deduplicate_qa(qa_pairs)

    ru_samples = [qa_to_chat(item) for item in qa_pairs]
    print(f"  RU Q&A после дедупликации: {len(ru_samples)}")

    print()
    print("=" * 55)
    print("Шаг 2. Загрузка и выборка coding примеров")
    print("=" * 55)

    df = load_coding_dataset()
    coding_df = sample_coding_balanced(df)

    # Конвертируем pandas DataFrame → список chat-словарей
    # row.to_dict() превращает строку датафрейма в обычный словарь
    # {"instruction": "...", "input": "...", "output": "..."} —
    # именно тот формат, который ожидает coding_to_chat()
    coding_samples = [
        coding_to_chat(row.to_dict()) for _, row in coding_df.iterrows()
    ]

    print()
    print("=" * 55)
    print("Шаг 3. Сборка финального датасета")
    print("=" * 55)

    all_samples = ru_samples + coding_samples
    print(f"  RU Q&A:  {len(ru_samples)}")
    print(f"  Coding:  {len(coding_samples)}")
    print(f"  Итого:   {len(all_samples)}")
    print(
        f"  Соотношение: {len(ru_samples) / len(all_samples) * 100:.0f}% RU / "
        f"{len(coding_samples) / len(all_samples) * 100:.0f}% coding"
    )

    random.seed(42)
    random.shuffle(all_samples)

    # Сплит 90/10
    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    val = all_samples[split_idx:]

    Path("data/final").mkdir(parents=True, exist_ok=True)
    save_jsonl(train, "data/final/train.jsonl")
    save_jsonl(val, "data/final/val.jsonl")

    print()
    print("=" * 55)
    print("Готово!")
    print("=" * 55)
    print(f"  Train: {len(train)} примеров → data/final/train.jsonl")
    print(f"  Val:   {len(val)} примеров   → data/final/val.jsonl")


if __name__ == "__main__":
    build_dataset()
