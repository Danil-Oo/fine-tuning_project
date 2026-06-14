"""
Модуль доменной оценки (LLM-архитектура) через LLM-as-a-judge.
Судья: deepseek/deepseek-chat через OpenRouter API.
"""

import json
import re
import time

import requests
import torch
from tqdm import tqdm

JUDGE_MODEL = "deepseek/deepseek-chat"
# JUDGE_MODEL = "stepfun/step-3.5-flash:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

JUDGE_SYSTEM_PROMPT = """Ты — эксперт-оценщик ответов языковых моделей. 
Твоя задача: оценить ответ языковой модели на вопрос по теме архитектуры и обучения LLM.

Оценивай по следующим критериям:
1. Техническая точность (правильность фактов, формул, концепций)
2. Полнота (все ли ключевые аспекты раскрыты)
3. Ясность изложения (структурированность, понятность)

Формат ответа — строго JSON:
{
  "score": <число от 1 до 5>,
  "reasoning": "<краткое обоснование на русском языке, 2-3 предложения>",
  "key_mistakes": ["<ошибка 1>", "<ошибка 2>"]  // пустой список если ошибок нет
}

Шкала:
5 — Отличный ответ: всё верно, полно, ясно
4 — Хороший ответ: верно, но неполно или недостаточно ясно
3 — Удовлетворительный: частично верно, есть пробелы
2 — Слабый ответ: есть существенные ошибки
1 — Неверный или нерелевантный ответ"""


def _call_judge(
    api_key: str,
    question: str,
    model_answer: str,
    reference_answer: str,
    retries: int = 5,
    timeout: int = 120,
) -> dict:
    """Вызов судьи через OpenRouter."""

    user_prompt = f"""**Вопрос:**
{question}

**Ответ модели:**
{model_answer}

**Эталонный ответ (для справки):**
{reference_answer}

Оцени ответ модели."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "reasoning": {"exclude": True},
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f" [Judge] Rate limit (429). Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning", "")
            # content = response.json()["choices"][0]["message"]["content"]

            if not content:
                raise ValueError("Both content and reasoning are empty")
            # Парсим JSON из ответа судьи
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"No JSON found in judge response: {content}")

        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (2**attempt)
                print(
                    f"  [Judge] Attempt {attempt + 1} failed: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(f"  [Judge] All retries failed: {e}")
                return {
                    "score": -1,
                    "reasoning": f"Judge error: {e}",
                    "key_mistakes": [],
                }


def generate_model_answer(
    model, tokenizer, question: str, device: str, max_new_tokens: int = 400
) -> str:
    """Генерирует ответ модели на вопрос."""
    messages = [
        {
            "role": "system",
            "content": "Ты — технический ассистент, эксперт по архитектуре и обучению языковых моделей. Отвечай на русском языке точно и по делу.",
        },
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_domain_evaluation(
    model,
    tokenizer,
    device: str,
    questions_path: str,
    api_key: str,
    n_questions: int = None,
) -> dict:
    """
    Запускает полную доменную оценку.

    Возвращает:
    {
      "avg_score": float,
      "n_evaluated": int,
      "results": [ { "id", "question", "model_answer", "score", "reasoning", "key_mistakes" }, ... ]
      "scores_by_category": { category: avg_score }
    }
    """
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    if n_questions:
        questions = questions[:n_questions]

    results = []
    category_scores: dict[str, list[int]] = {}

    print(f"\n[Domain Eval] Evaluating {len(questions)} questions...")

    for q in tqdm(questions, desc="Domain evaluation"):
        # 1. Генерируем ответ модели
        model_answer = generate_model_answer(
            model, tokenizer, q["question"], device
        )

        # 2. Судья оценивает
        verdict = _call_judge(
            api_key=api_key,
            question=q["question"],
            model_answer=model_answer,
            reference_answer=q["reference_answer"],
        )

        result = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "model_answer": model_answer,
            "reference_answer": q["reference_answer"],
            "score": verdict.get("score", -1),
            "reasoning": verdict.get("reasoning", ""),
            "key_mistakes": verdict.get("key_mistakes", []),
        }
        results.append(result)

        cat = q["category"]
        score = verdict.get("score", -1)
        if score > 0:
            category_scores.setdefault(cat, []).append(score)

        print(f"  [{q['id']}] Score: {score}/5 | {q['question'][:60]}...")

        # Небольшая пауза чтобы не превысить rate limit
        time.sleep(0.5)

    valid_scores = [r["score"] for r in results if r["score"] > 0]
    avg_score = (
        round(sum(valid_scores) / len(valid_scores), 3)
        if valid_scores
        else 0.0
    )

    scores_by_category = {
        cat: round(sum(scores) / len(scores), 3)
        for cat, scores in category_scores.items()
    }

    print(f"\n[Domain Eval] Average score: {avg_score:.3f}/5.0")

    return {
        "avg_score": avg_score,
        "n_evaluated": len(results),
        "results": results,
        "scores_by_category": scores_by_category,
    }
