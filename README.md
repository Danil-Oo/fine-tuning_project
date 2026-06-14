# QLoRA Fine-Tuning Qwen2.5-Coder-1.5B на домашнем GPU

Дообучение кодовой LLM на русскоязычных данных об архитектуре LLM с помощью QLoRA на GPU с 4GB VRAM (GTX 1650). Полный MLOps-пайплайн: от генерации данных до оценки модели.

---

## Проблема

Базовая модель [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) хорошо генерирует код (pass@1 = 76% на HumanEval), но плохо отвечает на вопросы по архитектуре LLM (2.5/5.0 по оценке LLM-судьи). Модель не знает про GQA, RoPE, KV-cache и другие концепции, которые не попали в её предобучение.

**Цель:** улучшить знания по LLM-архитектуре без потери кодовых способностей.

## Решение

- **QLoRA** (4-bit NF4 + LoRA r=16) — обучение на GPU с 4GB VRAM
- **Смешанный датасет:** 70% русскоязычных Q&A по архитектуре LLM + 30% английских coding tasks (для предотвращения catastrophic forgetting)
- **Модульная система оценки:** perplexity, domain Q&A (LLM-as-judge через DeepSeek), coding (unit tests + LLM judge), HumanEval Mini (pass@1)
- **Инфраструктура:** MLflow для трекинга экспериментов, Grafana + Prometheus для мониторинга GPU

## Результаты

| Метрика | Baseline | Run 01 (200 шагов) | Checkpoint 400 |
|---------|----------|-------------------|----------------|
| **Perplexity** | 7.21 | **2.94** | 3.01 |
| **Domain score** (LLM-as-judge) | 2.5 / 5.0 | 3.05 / 5.0 | **3.1 / 5.0** |
| **Coding pass rate** | 90% | 80% | **90%** |
| **Coding quality** (LLM judge) | 4.3 / 5.0 | 4.2 / 5.0 | **4.4 / 5.0** |
| **HumanEval pass@1** | 76% | 72% | **76%** |

### Анализ

Два чекпоинта показывают разные trade-offs:

- **Run 01 (200 шагов)** — лучшая perplexity (2.94), хорошее улучшение domain (+22%), но потеря в кодовых метриках (coding 90%→80%, HumanEval 76%→72%). Модель сфокусировалась на новых знаниях за счёт забытых кодовых паттернов.

- **Checkpoint 400** — perplexity чуть хуже (3.01 vs 2.94), но domain вырос до 3.1, а кодовые метрики полностью восстановлены (coding 90%, HumanEval 76%). Более продолжительное обучение позволило модели найти баланс между domain-знаниями и кодовыми способностями.

**Вывод:** оба чекпоинта đạtают цели проекта — domain score вырос с 2.5 до 3.05-3.1 (+22-24%). Выбор между ними зависит от приоритетов: если важна perplexity — Run 01, если важен баланс всех метрик — Checkpoint 400.

## Архитектура проекта

```
fine-tuning_project/
├── scripts/
│   ├── 01_generate_qa.py          # Генерация Q&A пар через OpenRouter API
│   ├── 02_prepare_dataset.py      # Подготовка датасета: дедупликация, сплит, конвертация
│   ├── 03_train.py                # QLoRA обучение (Unsloth + TRL SFTTrainer)
│   ├── 04_evaluate.py             # Оркестрация оценки: perplexity, domain, coding, HumanEval
│   ├── merge_lora.py              # Склейка LoRA адаптеров с базовой моделью
│   ├── merge_qa.py                # Утилита для добавления Q&A пар в датасет
│   ├── rerun_coding_judge.py      # Перезапуск LLM-судьи для_failed оценок
│   └── eval/
│       ├── perplexity.py          # Вычисление perplexity на валидационном сете
│       ├── domain_eval.py         # Оценка знаний по LLM (LLM-as-judge)
│       ├── coding_eval.py         # Оценка кода (unit tests + LLM-as-judge)
│       └── humaneval_mini.py      # Mini-HumanEval: 25 задач, pass@1
├── configs/
│   ├── training_config.yaml       # Гиперпараметры обучения (QLoRA, scheduler, optimizer)
│   └── Modelfile                  # Ollama modelfile для сервинга GGUF модели
├── data/
│   ├── generated/qa_pairs_ru.json # Сгенерированные Q&A пары
│   ├── final/
│   │   ├── train.jsonl            # Тренировочный датасет (1105 примеров)
│   │   └── val.jsonl              # Валидационный датасет (123 примера)
│   └── eval/
│       ├── coding_tasks.json      # Задачи для coding eval
│       └── domain_questions.json  # Вопросы для domain eval
├── jupyter-notebooks/
│   ├── code_dataset_exploration.ipynb   # EDA кодового датасета
│   ├── tokenization_stats.ipynb         # Анализ токенизации и длин последовательностей
│   └── testing_evaluation_code.ipynb    # Тестирование модулей оценки
├── monitoring/
│   ├── docker-compose.yml         # Prometheus + Grafana + nvidia-exporter
│   ├── gpu_monitor.py             # Flask API для GPU метрик
│   └── prometheus/prometheus.yml  # Конфигурация Prometheus
└── .env.example                   # Шаблон переменных окружения
```

## Стек технологий

| Компонент | Технология |
|-----------|-----------|
| **Модель** | Qwen2.5-Coder-1.5B-Instruct |
| **Fine-tuning** | Unsloth + TRL (SFTTrainer) + PEFT (LoRA) |
| **Квантизация** | 4-bit NF4 с double quantization |
| **Оптимизатор** | paged_adamw_8bit |
| **Precision** | bfloat16 |
| **Трекинг экспериментов** | MLflow (SQLite backend) |
| **Мониторинг GPU** | Grafana + Prometheus + nvidia_gpu_exporter |
| **Оценка** | perplexity, LLM-as-judge (DeepSeek), HumanEval Mini |
| **Датасет** | ShareGPT/chat format, 1228 примеров |

## Быстрый старт

### 1. Установка зависимостей

```bash
# Клонировать репозиторий
git clone https://github.com/Danil-Oo/fine-tuning_project.git
cd fine-tuning_project

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

```bash
cp .env.example .env
# Отредактировать .env и вставить реальный API ключ
```

| Переменная | Описание | Где получить |
|-----------|----------|-------------|
| `OPENROUTER_API_KEY` | API ключ для генерации Q&A и LLM-as-judge | [OpenRouter](https://openrouter.ai/) |
| `MODEL_PATH` | Путь к базовой модели | По умолчанию: `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| `OLLAMA_MODEL_NAME` | Имя модели в Ollama | Произвольное |

### 3. Запуск пайплайна

```bash
# Шаг 1: Генерация Q&A пар (опционально — данные уже в data/generated/)
python scripts/01_generate_qa.py

# Шаг 2: Подготовка датасета
python scripts/02_prepare_dataset.py

# Шаг 3: Обучение
python scripts/03_train.py

# Шаг 4: Оценка
python scripts/04_evaluate.py --run-name "after_finetuning"

# Шаг 5: Склейка LoRA с базовой моделью
python scripts/merge_lora.py

# Шаг 6: Конвертация в GGUF для Ollama
# (требуется llama.cpp — clone и сборка отдельно)
```

### 4. Запуск мониторинга

```bash
# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

# GPU мониторинг (Prometheus + Grafana)
cd monitoring && docker compose up -d
```

## Структура данных

### Формат

Датасет использует ShareGPT/chat format с тремя ролями:

```json
{
  "conversations": [
    {"role": "system", "content": "Ты -- технический ассистент, специализирующийся на архитектуре больших языковых моделей..."},
    {"role": "user", "content": "Что такое механизм внимания (attention) в трансформерах?"},
    {"role": "assistant", "content": "Механизм внимания (attention) позволяет модели..."}
  ]
}
```

### Источники данных

- **70% — русскоязычные Q&A** по архитектуре LLM: сгенерированы из 70 заметок Obsidian (Attention, GQA, RoPE, KV-cache, Speculative Decoding, LoRA/QLoRA, RLHF/PPO/DPO, Transformer, FFN, SwiGLU, RMSNorm, MoE, Tokenization и др.)
- **30% — английские coding tasks** из датасета [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) (стратифицированная выборка)

### Генерация своих Q&A пар

```bash
# Скрипт генерирует Q&A пары из .md файлов через OpenRouter API
# Поддерживает возобновление при сбоях
python scripts/01_generate_qa.py
```

## Эксперименты

### Конфигурация LoRA

| Параметр | Значение |
|----------|----------|
| Rank (r) | 16 |
| Alpha | 32 (2x rank) |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Gradient checkpointing | unsloth mode |

### Гиперпараметры обучения

| Параметр | Значение |
|----------|----------|
| Epochs | 3 |
| Learning rate | 2e-4 |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Effective batch size | 8 (2 per-device x 4 gradient accumulation) |
| Max seq length | 2048 |
| Optimizer | paged_adamw_8bit |
| Precision | bf16 |

### Результаты по чекпоинтам

| Чекпоинт | Perplexity | Domain | Coding pass | Coding quality | HumanEval |
|----------|-----------|--------|-------------|----------------|-----------|
| Baseline (до обучения) | 7.21 | 2.5 | 90% | 4.3 | 76% |
| Run 01 (200 шагов) | **2.94** | 3.05 | 80% | 4.2 | 72% |
| Checkpoint 400 | 3.01 | **3.1** | **90%** | **4.4** | **76%** |

## Мониторинг и инфраструктура

### MLflow

- Трекинг метрик, гиперпараметров и артефактов
- Сравнение экспериментов (baseline vs разные чекпоинты обучения)
- SQLite backend для простоты (без внешней БД)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
# UI доступен на http://localhost:5000
```

### GPU мониторинг

- **Prometheus** — сбор метрик GPU (VRAM, utilization, temperature)
- **Grafana** — визуализация и алерты
- **nvidia_gpu_exporter** — экспорт метрик NVIDIA GPU

```bash
cd monitoring && docker compose up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Модульная система оценки

Оценка включает 4 независимых модуля:

1. **Perplexity** — непрерывная метрика качества языковой модели
2. **Domain Eval** — LLM-as-judge (DeepSeek) оценивает ответы по архитектуре LLM по шкале 1-5
3. **Coding Eval** — генерация кода + запуск unit tests + LLM-as-judge для оценки качества
4. **HumanEval Mini** — 25 задач из HumanEval, измерение pass@1

## Возможные улучшения

- [ ] Эксперименты с разным rank LoRA (r=8, r=32)
- [ ] Увеличение числа эпох с early stopping
- [ ] Использование SWE-bench для coding данных
- [ ] RAG/Agent система на базе дообученной модели
- [ ] Сервинг через Ollama + GGUF квантизация

## Лицензия

MIT
