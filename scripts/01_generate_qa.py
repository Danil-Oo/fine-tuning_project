"""
01_generate_qa.py
Генерация Q&A пар из .md файлов через OpenRouter API.
Модель: deepseek/deepseek-chat-v3-0324 (дёшево + хорошее качество RU)
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI  # OpenRouter совместим с openai SDK

load_dotenv(override=True)

client = OpenAI(
    # api_key=os.environ["OPENROUTER_API_KEY"],
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "deepseek/deepseek-chat-v3-0324"  # меняй здесь при необходимости

SYSTEM_PROMPT = """Ты — эксперт по архитектуре больших языковых моделей.
На основе предоставленного текста сгенерируй {n_questions} вопросов и развёрнутых ответов.

Вопросы должны быть разных типов:
- Определения ("Что такое X?")
- Механизмы ("Как работает X?")
- Сравнения ("В чём разница между X и Y?")
- Применение ("Когда использовать X вместо Y?")
- Причины ("Почему в X используется подход Y?")

Требования к ответам:
- 4–7 предложений, не более 500 слов
- Технически точные, на русском языке
- Самодостаточные (не ссылаются на "текст выше")

Верни ТОЛЬКО валидный JSON массив без пояснений и без markdown-обёрток:
[
  {{"question": "...", "answer": "..."}},
  ...
]"""


def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    """Разбивает длинный текст на куски по абзацам."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def generate_qa_from_text(text: str, n_questions: int = 15) -> list[dict]:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": SYSTEM_PROMPT.format(n_questions=n_questions)
                + f"\n\nТекст:\n{text}",
            }
        ],
        extra_headers={
            "HTTP-Referer": "https://github.com/llm-finetuning",  # опционально
        },
    )
    raw = response.choices[0].message.content.strip()
    # Убираем возможные markdown-обёртки
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    return json.loads(raw)


def process_knowledge_base(raw_dir: str, output_dir: str):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Загружаем уже сгенерированное (для возобновления при сбое)
    output_file = output_path / "qa_pairs_ru.json"
    all_pairs = []
    processed_files = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            all_pairs = json.load(f)
        processed_files = {p["source_file"] for p in all_pairs}
        print(
            f"Возобновление: уже есть {len(all_pairs)} пар из {len(processed_files)} файлов"
        )

    md_files = sorted(raw_path.rglob("*.md"))
    print(f"Всего файлов: {len(md_files)}")

    for md_file in md_files:
        if md_file.name in processed_files:
            print(f"Пропускаю (уже обработан): {md_file.name}")
            continue

        content = md_file.read_text(encoding="utf-8")
        if len(content) < 300:
            print(f"Пропускаю (слишком короткий): {md_file.name}")
            continue

        chunks = chunk_text(content, max_chars=6000)
        # Распределяем вопросы по чанкам
        questions_per_chunk = max(5, 15 // len(chunks))

        print(
            f"\nОбрабатываю: {md_file.name} ({len(content)} символов, {len(chunks)} чанков)"
        )

        for i, chunk in enumerate(chunks):
            try:
                pairs = generate_qa_from_text(
                    chunk, n_questions=questions_per_chunk
                )
                for pair in pairs:
                    pair["source_file"] = md_file.name
                    pair["source_section"] = str(md_file.parent.name)
                    pair["chunk_index"] = i
                all_pairs.extend(pairs)
                print(
                    f"  Чанк {i + 1}/{len(chunks)}: сгенерировано {len(pairs)} пар"
                )

                # Сохраняем после каждого чанка (защита от сбоев)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_pairs, f, ensure_ascii=False, indent=2)

                time.sleep(1)  # Пауза между запросами

            except json.JSONDecodeError as e:
                print(f"  Ошибка парсинга JSON: {e}")
            except Exception as e:
                print(f"  Ошибка API: {e}")
                time.sleep(5)

    print(f"\n{'=' * 50}")
    print(f"Итого Q&A пар: {len(all_pairs)}")
    print(f"Сохранено в: {output_file}")


if __name__ == "__main__":
    process_knowledge_base("data/raw", "data/generated")


"""
Ключевые улучшения по сравнению с оригиналом:
- **Возобновление** — если скрипт упал, он продолжит с того места где остановился
- **Chunking** — большие файлы (100k слов) автоматически режутся на куски по 6000 символов
- **Сохранение после каждого чанка** — не потеряешь прогресс
- **Пауза между запросами** — не словишь rate limit

---

## 3. Гибридный подход: чат + API

Это разумная стратегия. Вот как я бы это организовал:

**Для генерации через чат-интерфейсы:**

Подготовь шаблон промпта, который будешь вставлять вместе с текстом заметки:
```
Ты — эксперт по архитектуре больших языковых моделей.
На основе текста ниже сгенерируй 15 вопросов и развёрнутых ответов.

Типы вопросов: определения, механизмы, сравнения, применение, причины.
Требования к ответам: 4–7 предложений, ≤500 слов, технически точно, на русском.

Верни ТОЛЬКО JSON массив:
[{"question": "...", "answer": "..."}, ...]

Текст:
[ВСТАВЬ ТЕКСТ ЗАМЕТКИ СЮДА]
"""
