"""
Mini-HumanEval — оценка pass@1 на подмножестве задач HumanEval.
Использует 25 репрезентативных задач.
Задачи встроены в скрипт, внешних зависимостей на evalplus не требует.
"""

import ast
import re
import traceback

import torch
from tqdm import tqdm

# 25 задач из HumanEval, отобранных по репрезентативности и уровню сложности
HUMANEVAL_TASKS = [
    {
        "task_id": "HE_1",
        "prompt": 'def has_close_elements(numbers: list, threshold: float) -> bool:\n    """Check if any two numbers in the list are closer to each other than the given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "test": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HE_2",
        "prompt": "def separate_paren_groups(paren_string: str) -> list:\n    \"\"\"Separate a string of multiple groups of nested parentheses into a list of strings.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "test": "assert separate_paren_groups('(()()) ((())) () ((())(()))') == ['(()())', '((()))', '()', '((())(()))']",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HE_3",
        "prompt": 'def truncate_number(number: float) -> float:\n    """Return the decimal part of the float number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
        "test": "assert truncate_number(3.5) == 0.5\nassert abs(truncate_number(1.33) - 0.33) < 1e-6\nassert abs(truncate_number(123.456) - 0.456) < 1e-6",
        "entry_point": "truncate_number",
    },
    {
        "task_id": "HE_4",
        "prompt": 'def below_zero(operations: list) -> bool:\n    """Given a list of deposit and withdrawal operations on a bank account starting at zero, detect if at any point the balance falls below zero.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    """\n',
        "test": "assert below_zero([]) == False\nassert below_zero([1, 2, -3, 1, 2, -3]) == False\nassert below_zero([1, 2, -4, 5, 6]) == True\nassert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) == False",
        "entry_point": "below_zero",
    },
    {
        "task_id": "HE_5",
        "prompt": 'def mean_absolute_deviation(numbers: list) -> float:\n    """Calculate Mean Absolute Deviation around the mean of the list.\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """\n',
        "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\nassert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6",
        "entry_point": "mean_absolute_deviation",
    },
    {
        "task_id": "HE_6",
        "prompt": 'def intersperse(numbers: list, delimeter: int) -> list:\n    """Insert a number \'delimeter\' between every two consecutive elements of the list.\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    """\n',
        "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]\nassert intersperse([1, 2, 3, 4, 5], 8) == [1, 8, 2, 8, 3, 8, 4, 8, 5]",
        "entry_point": "intersperse",
    },
    {
        "task_id": "HE_7",
        "prompt": 'def parse_nested_parens(paren_string: str) -> list:\n    """For each group of nested parentheses, return the deepest nesting level.\n    >>> parse_nested_parens(\'(()()) ((())) () ((())(())())\')\n    [2, 3, 1, 3]\n    """\n',
        "test": "assert parse_nested_parens('(()()) ((())) () ((())(()))') == [2, 3, 1, 3]",
        "entry_point": "parse_nested_parens",
    },
    {
        "task_id": "HE_8",
        "prompt": "def filter_by_substring(strings: list, substring: str) -> list:\n    \"\"\"Filter a list of strings to only those containing the given substring.\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n",
        "test": "assert filter_by_substring([], 'john') == []\nassert filter_by_substring(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'XXX'], 'xxx') == ['xxx', 'xxy', 'xxxAAA']\nassert filter_by_substring(['grunt', 'trumpet', 'prune', 'gruesome'], 'run') == ['grunt', 'prune']",
        "entry_point": "filter_by_substring",
    },
    {
        "task_id": "HE_9",
        "prompt": 'def sum_product(numbers: list) -> tuple:\n    """Return a tuple of sum and product of all the integers in the list.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    """\n',
        "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 1, 1]) == (3, 1)\nassert sum_product([100, 0]) == (100, 0)\nassert sum_product([3, 5, 7]) == (15, 105)",
        "entry_point": "sum_product",
    },
    {
        "task_id": "HE_10",
        "prompt": 'def rolling_max(numbers: list) -> list:\n    """Return a list of rolling maximum values from the input list.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    """\n',
        "test": "assert rolling_max([]) == []\nassert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]\nassert rolling_max([4, 3, 2, 1]) == [4, 4, 4, 4]",
        "entry_point": "rolling_max",
    },
    {
        "task_id": "HE_11",
        "prompt": "def is_palindrome(string: str) -> bool:\n    \"\"\"Check if a string is a palindrome.\n    >>> is_palindrome('')\n    True\n    >>> is_palindrome('aba')\n    True\n    >>> is_palindrome('zbcd')\n    False\n    \"\"\"\n",
        "test": "assert is_palindrome('') == True\nassert is_palindrome('aba') == True\nassert is_palindrome('zbcd') == False\nassert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False",
        "entry_point": "is_palindrome",
    },
    {
        "task_id": "HE_12",
        "prompt": "def make_palindrome(string: str) -> str:\n    \"\"\"Find the shortest palindrome that begins with the supplied string. The algorithm is: find the longest postfix of the string that is a palindrome, append the reverse of the string prefix to the end.\n    >>> make_palindrome('')\n    ''\n    >>> make_palindrome('cat')\n    'catac'\n    >>> make_palindrome('cata')\n    'catac'\n    \"\"\"\n",
        "test": "assert make_palindrome('') == ''\nassert make_palindrome('x') == 'x'\nassert make_palindrome('xyz') == 'xyzyx'\nassert make_palindrome('cat') == 'catac'",
        "entry_point": "make_palindrome",
    },
    {
        "task_id": "HE_13",
        "prompt": "def string_xor(a: str, b: str) -> str:\n    \"\"\"Perform XOR on two strings of binary digits and return the result.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
        "test": "assert string_xor('111000', '101010') == '010010'\nassert string_xor('1', '1') == '0'\nassert string_xor('0101', '0000') == '0101'",
        "entry_point": "string_xor",
    },
    {
        "task_id": "HE_14",
        "prompt": "def longest(strings: list) -> str | None:\n    \"\"\"Return the longest string from a list. If multiple have the same length, return the first. Return None for empty list.\n    >>> longest([])\n    \n    >>> longest(['a', 'b', 'c'])\n    'a'\n    >>> longest(['a', 'bb', 'ccc'])\n    'ccc'\n    \"\"\"\n",
        "test": "assert longest([]) is None\nassert longest(['a', 'b', 'c']) == 'a'\nassert longest(['a', 'bb', 'ccc']) == 'ccc'\nassert longest(['abc', 'def', 'g']) == 'abc'",
        "entry_point": "longest",
    },
    {
        "task_id": "HE_15",
        "prompt": 'def greatest_common_divisor(a: int, b: int) -> int:\n    """Return the GCD of two integers.\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 15)\n    5\n    """\n',
        "test": "assert greatest_common_divisor(3, 7) == 1\nassert greatest_common_divisor(10, 15) == 5\nassert greatest_common_divisor(49, 14) == 7\nassert greatest_common_divisor(144, 60) == 12",
        "entry_point": "greatest_common_divisor",
    },
    {
        "task_id": "HE_16",
        "prompt": "def all_prefixes(string: str) -> list:\n    \"\"\"Return a list of all prefixes of the string, from shortest to longest.\n    >>> all_prefixes('abc')\n    ['a', 'ab', 'abc']\n    \"\"\"\n",
        "test": "assert all_prefixes('') == []\nassert all_prefixes('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']\nassert all_prefixes('www') == ['w', 'ww', 'www']",
        "entry_point": "all_prefixes",
    },
    {
        "task_id": "HE_17",
        "prompt": 'def string_sequence(n: int) -> str:\n    """Return a string with space-delimited numbers starting from 0 to n.\n    >>> string_sequence(0)\n    \'0\'\n    >>> string_sequence(5)\n    \'0 1 2 3 4 5\'\n    """\n',
        "test": "assert string_sequence(0) == '0'\nassert string_sequence(3) == '0 1 2 3'\nassert string_sequence(10) == '0 1 2 3 4 5 6 7 8 9 10'",
        "entry_point": "string_sequence",
    },
    {
        "task_id": "HE_18",
        "prompt": 'def count_distinct_characters(string: str) -> int:\n    """Return the count of distinct characters in a string, case insensitive.\n    >>> count_distinct_characters(\'xyzXYZ\')\n    3\n    >>> count_distinct_characters(\'Jerry\')\n    4\n    """\n',
        "test": "assert count_distinct_characters('') == 0\nassert count_distinct_characters('abcde') == 5\nassert count_distinct_characters('abcde' + 'ABCDE') == 5\nassert count_distinct_characters('aabbcc') == 3",
        "entry_point": "count_distinct_characters",
    },
    {
        "task_id": "HE_19",
        "prompt": "def parse_music(music_string: str) -> list:\n    \"\"\"Parse a music string where 'o' = 4 beats, 'o|' = 2 beats, '.|' = 1 beat. Return a list of beat counts.\n    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    \"\"\"\n",
        "test": "assert parse_music('') == []\nassert parse_music('o o o o') == [4, 4, 4, 4]\nassert parse_music('.| .| .| .|') == [1, 1, 1, 1]\nassert parse_music('o| o| .| .| o o') == [2, 2, 1, 1, 4, 4]",
        "entry_point": "parse_music",
    },
    {
        "task_id": "HE_20",
        "prompt": "def how_many_times(string: str, substring: str) -> int:\n    \"\"\"Count how many times a substring appears in a string, including overlapping occurrences.\n    >>> how_many_times('', 'a')\n    0\n    >>> how_many_times('aaa', 'a')\n    3\n    >>> how_many_times('aaaa', 'aa')\n    3\n    \"\"\"\n",
        "test": "assert how_many_times('', 'x') == 0\nassert how_many_times('aaaa', 'aa') == 3\nassert how_many_times('aaa', 'a') == 3\nassert how_many_times('abcabcabc', 'abc') == 3",
        "entry_point": "how_many_times",
    },
    {
        "task_id": "HE_21",
        "prompt": "def sort_numbers(numbers: str) -> str:\n    \"\"\"Sort a space-separated string of spelled-out numbers ('zero' to 'nine').\n    >>> sort_numbers('three one five')\n    'one three five'\n    \"\"\"\n",
        "test": "assert sort_numbers('') == ''\nassert sort_numbers('three') == 'three'\nassert sort_numbers('three one five') == 'one three five'\nassert sort_numbers('nine eight seven six five four three two one zero') == 'zero one two three four five six seven eight nine'",
        "entry_point": "sort_numbers",
    },
    {
        "task_id": "HE_22",
        "prompt": 'def find_closest_elements(numbers: list) -> tuple:\n    """From a list of floats, return the two closest values as a sorted tuple.\n    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])\n    (2.0, 2.2)\n    """\n',
        "test": "assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)\nassert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)",
        "entry_point": "find_closest_elements",
    },
    {
        "task_id": "HE_23",
        "prompt": 'def rescale_to_unit(numbers: list) -> list:\n    """Rescale a list of numbers to the [0, 1] interval using min-max normalization.\n    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n    [0.0, 0.25, 0.5, 0.75, 1.0]\n    """\n',
        "test": "assert rescale_to_unit([2.0, 49.9]) == [0.0, 1.0]\nassert rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]",
        "entry_point": "rescale_to_unit",
    },
    {
        "task_id": "HE_24",
        "prompt": 'def filter_integers(values: list) -> list:\n    """Filter only integers from a list of mixed types.\n    >>> filter_integers([\'a\', 3.14, 5])\n    [5]\n    >>> filter_integers([1, 2, 3, \'abc\', {}, []])\n    [1, 2, 3]\n    """\n',
        "test": "assert filter_integers([]) == []\nassert filter_integers([1, 2, 3]) == [1, 2, 3]\nassert filter_integers(['a', 3.14, 5]) == [5]\nassert filter_integers([1, 2, 3, 'abc', {}, []]) == [1, 2, 3]",
        "entry_point": "filter_integers",
    },
    {
        "task_id": "HE_25",
        "prompt": 'def strlen(string: str) -> int:\n    """Return the length of the given string.\n    >>> strlen(\'\')\n    0\n    >>> strlen(\'abc\')\n    3\n    """\n',
        "test": "assert strlen('') == 0\nassert strlen('x') == 1\nassert strlen('asdfgh') == 6",
        "entry_point": "strlen",
    },
]


def _generate_completion(model, tokenizer, prompt: str, device: str) -> str:
    """Генерирует completion для HumanEval промпта."""
    messages = [
        {
            "role": "system",
            "content": "Complete the Python function. Return only the function implementation, no explanation.",
        },
        {
            "role": "user",
            "content": f"Complete this Python function:\n\n```python\n{prompt}\n```\n\nReturn only the complete function with the implementation.",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _extract_function_from_response(
    response: str, prompt: str, entry_point: str
) -> str:
    """Извлекает и собирает полный код функции из ответа."""
    # Ищем полную функцию в ответе
    code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
        if f"def {entry_point}" in code:
            return code

    # Если нет обёртки, ищем тело функции
    lines = response.split("\n")
    func_lines = []
    in_func = False
    for line in lines:
        if f"def {entry_point}" in line:
            in_func = True
        if in_func:
            func_lines.append(line)

    if func_lines:
        return "\n".join(func_lines)

    # Fallback: prompt + ответ как тело функции
    return prompt + "\n" + response


def _run_humaneval_test(
    full_code: str, test_code: str, entry_point: str
) -> dict:
    """Запускает тест для HumanEval задачи."""
    try:
        ast.parse(full_code)
    except SyntaxError as e:
        return {"passed": False, "error": str(e), "error_type": "SyntaxError"}

    namespace = {}
    try:
        exec(full_code, namespace)
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }

    if entry_point not in namespace:
        return {
            "passed": False,
            "error": f"Function '{entry_point}' not found",
            "error_type": "NotFound",
        }

    # Инжектируем функцию в глобальный namespace и запускаем тест
    test_namespace = {entry_point: namespace[entry_point]}
    try:
        exec(test_code, test_namespace)
        return {"passed": True, "error": None, "error_type": None}
    except AssertionError:
        return {
            "passed": False,
            "error": traceback.format_exc(),
            "error_type": "AssertionError",
        }
    except Exception as e:
        return {
            "passed": False,
            "error": traceback.format_exc(),
            "error_type": type(e).__name__,
        }


def run_humaneval_mini(model, tokenizer, device: str) -> dict:
    """
    Запускает Mini-HumanEval на 25 задачах.

    Возвращает:
    {
      "pass_at_1": float,
      "n_tasks": int,
      "n_passed": int,
      "results": [ {...} ]
    }
    """
    results = []
    print(f"\n[HumanEval Mini] Running {len(HUMANEVAL_TASKS)} tasks...")

    for task in tqdm(HUMANEVAL_TASKS, desc="HumanEval"):
        # Генерируем completion
        response = _generate_completion(
            model, tokenizer, task["prompt"], device
        )

        # Извлекаем и собираем код
        full_code = _extract_function_from_response(
            response, task["prompt"], task["entry_point"]
        )

        # Запускаем тест
        test_result = _run_humaneval_test(
            full_code, task["test"], task["entry_point"]
        )

        result = {
            "task_id": task["task_id"],
            "entry_point": task["entry_point"],
            "passed": test_result["passed"],
            "error_type": test_result.get("error_type"),
            "generated_code": full_code,
        }
        results.append(result)

        status = "✓" if test_result["passed"] else "✗"
        print(f"  {status} {task['task_id']} ({task['entry_point']})")

    n_passed = sum(1 for r in results if r["passed"])
    pass_at_1 = round(n_passed / len(results), 4) if results else 0.0

    print(
        f"\n[HumanEval Mini] pass@1 = {pass_at_1:.4f} ({n_passed}/{len(results)})"
    )

    return {
        "pass_at_1": pass_at_1,
        "n_tasks": len(results),
        "n_passed": n_passed,
        "results": results,
    }
