"""
Модуль оценки навыков написания кода.
- Unit-тесты: бинарный pass/fail
- LLM-as-a-judge: качество кода (читаемость, идиоматичность, edge cases)
"""

import ast
import json
import re
import time
import traceback

import requests
import torch
from tqdm import tqdm

# JUDGE_MODEL = "stepfun/step-3.5-flash:free"
JUDGE_MODEL = "deepseek/deepseek-chat"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

CODING_JUDGE_SYSTEM_PROMPT = """You are an expert Python code reviewer. 
Evaluate the provided code solution for a coding problem.

Focus on:
1. Code quality (readability, naming, style)
2. Pythonic idioms (use of built-ins, comprehensions, standard library)
3. Edge case handling (empty input, boundary values)
4. Time/space complexity (is there a clearly better approach?)

Respond strictly in JSON format:
{
  "quality_score": <integer 1-5>,
  "is_pythonic": <true/false>,
  "handles_edge_cases": <true/false>,
  "complexity_note": "<O(?) time, O(?) space — brief note>",
  "review": "<2-3 sentences in English>"
}

Scale:
5 — Clean, idiomatic, production-ready
4 — Good code, minor style issues
3 — Works but not idiomatic or has style issues
2 — Functional but poor quality
1 — Hard to read or maintain"""


def _extract_code_from_response(response_text: str) -> str:
    """Извлекает Python-код из ответа модели."""
    # Ищем ```python ... ``` блок
    match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Ищем ``` ... ``` без явного языка
    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Проверяем что это похоже на Python (содержит def или class)
        if "def " in code or "class " in code:
            return code

    # Ищем def/class в тексте без обёртки
    lines = response_text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("def ", "class ")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()
    else:
        print(
            "  [Warning] Could not extract code from response, returning raw text"
        )
        return response_text.strip()


def _run_unit_tests(code: str, test_code: str, task_id: str) -> dict:
    """
    Запускает unit-тесты для сгенерированного кода.

    Возвращает:
    {
      "passed": bool,
      "error": str or None,
      "error_type": str or None
    }
    """
    # Синтаксическая проверка
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"passed": False, "error": str(e), "error_type": "SyntaxError"}

    # Подготовка пространства имён
    namespace = {}

    try:
        exec(code, namespace)
    except Exception as e:
        return {
            "passed": False,
            "error": traceback.format_exc(),
            "error_type": type(e).__name__,
        }

    # Определяем имя функции/класса из test_code
    # test_code принимает fn или cls — нужно найти главный объект
    fn_name = None
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_") and name != "run_tests":
            fn_name = name

    if fn_name is None:
        return {
            "passed": False,
            "error": "No callable found in generated code",
            "error_type": "NotFound",
        }

    try:
        exec(test_code, namespace)
        run_tests = namespace["run_tests"]
        result = run_tests(namespace[fn_name])
        return {"passed": bool(result), "error": None, "error_type": None}
    except AssertionError:
        return {
            "passed": False,
            "error": f"AssertionError: {traceback.format_exc()}",
            "error_type": "AssertionError",
        }
    except Exception as e:
        return {
            "passed": False,
            "error": traceback.format_exc(),
            "error_type": type(e).__name__,
        }


def _call_coding_judge(
    api_key: str,
    task_prompt: str,
    code: str,
    passed_tests: bool,
    retries: int = 5,
) -> dict:
    """Вызов судьи для оценки качества кода."""
    user_prompt = f"""**Task:**
{task_prompt}

**Solution:**
```python
{code}
```

**Unit tests:** {"PASSED ✓" if passed_tests else "FAILED ✗"}

Evaluate the code quality."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": CODING_JUDGE_SYSTEM_PROMPT},
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
                timeout=60,
            )

            if response.status_code == 429:
                wait = 30 * (attempt + 1)  # 30, 60, 90, 120, 150 сек
                print(f"  [Judge] Rate limit (429). Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning", "")
            # content = response.json()["choices"][0]["message"]["content"]

            if not content:
                raise ValueError("Both content and reasoning are empty")

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"No JSON in judge response: {content}")

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (2**attempt))
            else:
                return {
                    "quality_score": -1,
                    "is_pythonic": False,
                    "handles_edge_cases": False,
                    "complexity_note": "N/A",
                    "review": f"Judge error: {e}",
                }


def generate_code_solution(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 600,
) -> str:
    """Генерирует решение задачи на код."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert Python developer. Provide clean, correct Python code solutions. Always wrap your code in ```python ... ``` blocks.",
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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


def run_coding_evaluation(
    model,
    tokenizer,
    device: str,
    tasks_path: str,
    api_key: str,
    n_tasks: int = None,
) -> dict:
    """
    Запускает полную оценку навыков кодирования.

    Возвращает:
    {
      "pass_rate": float,
      "avg_quality_score": float,
      "n_tasks": int,
      "results": [ {...} ]
      "pass_rate_by_difficulty": { difficulty: rate }
    }
    """
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    if n_tasks:
        tasks = tasks[:n_tasks]

    results = []
    difficulty_results: dict[str, list[bool]] = {}

    print(f"\n[Coding Eval] Evaluating {len(tasks)} tasks...")

    for task in tqdm(tasks, desc="Coding evaluation"):
        # 1. Генерируем решение
        raw_response = generate_code_solution(
            model, tokenizer, task["prompt"], device
        )
        code = _extract_code_from_response(raw_response)

        # 2. Запускаем unit-тесты
        test_result = _run_unit_tests(code, task["test_code"], task["id"])

        # 3. LLM-as-a-judge для качества кода
        judge_result = _call_coding_judge(
            api_key=api_key,
            task_prompt=task["prompt"],
            code=code,
            passed_tests=test_result["passed"],
        )

        result = {
            "id": task["id"],
            "difficulty": task["difficulty"],
            "category": task["category"],
            "prompt": task["prompt"],
            "generated_code": code,
            "passed": test_result["passed"],
            "error": test_result.get("error"),
            "error_type": test_result.get("error_type"),
            "quality_score": judge_result.get("quality_score", -1),
            "is_pythonic": judge_result.get("is_pythonic", False),
            "handles_edge_cases": judge_result.get(
                "handles_edge_cases", False
            ),
            "complexity_note": judge_result.get("complexity_note", ""),
            "review": judge_result.get("review", ""),
        }
        results.append(result)

        diff = task["difficulty"]
        difficulty_results.setdefault(diff, []).append(test_result["passed"])

        status = "✓ PASS" if test_result["passed"] else "✗ FAIL"
        print(
            f"  [{task['id']}] {status} | Quality: {result['quality_score']}/5 | {task['category']}"
        )

        time.sleep(0.5)

    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = round(passed_count / len(results), 3) if results else 0.0

    valid_quality = [
        r["quality_score"] for r in results if r["quality_score"] > 0
    ]
    avg_quality = (
        round(sum(valid_quality) / len(valid_quality), 3)
        if valid_quality
        else 0.0
    )

    pass_rate_by_difficulty = {
        diff: round(sum(passed) / len(passed), 3)
        for diff, passed in difficulty_results.items()
    }

    print(
        f"\n[Coding Eval] Pass rate: {pass_rate:.1%} | Avg quality: {avg_quality:.2f}/5"
    )

    return {
        "pass_rate": pass_rate,
        "avg_quality_score": avg_quality,
        "n_tasks": len(results),
        "results": results,
        "pass_rate_by_difficulty": pass_rate_by_difficulty,
    }
