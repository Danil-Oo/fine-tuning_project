"""
Основной скрипт QLoRA fine-tuning.
Запускай из корня проекта: python scripts/03_train.py
"""

# ── Стандартные библиотеки ────────────────────────────────────────────────────

from pathlib import Path  # noqa: I001

import torch
import yaml

# ── HuggingFace экосистема ────────────────────────────────────────────────────
from datasets import load_dataset


from transformers import TrainerCallback


# ── Unsloth ───────────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel
# FastLanguageModel — главный класс Unsloth.
# Это "умная обёртка" над HuggingFace Transformers, которая:
# 1) Вызывает BitsAndBytes для NF4-квантизации при загрузке модели
# 2) Подменяет стандартные attention-ядра своими Triton-оптимизированными
#    (отсюда -40-70% VRAM и ускорение в 2-5x)
# 3) Через get_peft_model добавляет LoRA-адаптеры поверх замороженных весов

# ── TRL ───────────────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer
# SFTTrainer (Supervised Fine-Tuning Trainer) — специализированный trainer
# для обучения LLM на инструкциях/диалогах. Надстройка над Trainer из Transformers.
# Умеет: токенизировать текст на лету, правильно маскировать токены промпта
# (чтобы loss считался только по ответу ассистента), работать с LoRA-моделями.
#
# SFTConfig — конфиг для SFTTrainer. Наследуется от TrainingArguments
# (стандартный класс Transformers) и добавляет SFT-специфичные параметры:
# dataset_text_field, max_seq_length, packing и др.

import mlflow

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
mlflow.set_tracking_uri(f"sqlite:///{_PROJECT_ROOT}/mlflow/mlflow.db")

# =============================================================================
# 1. ЗАГРУЗКА КОНФИГУРАЦИИ
# =============================================================================

with open(
    "/home/danil-pc/fine-tuning_project/configs/training_config.yaml"
) as f:
    cfg = yaml.safe_load(f)


print(f"Конфиг загружен: {len(cfg)} параметров")
print(f"Модель: {cfg['model_name']}")
print(f"max_seq_length: {cfg['max_seq_length']}")

# =============================================================================
# 2. ЗАГРУЗКА МОДЕЛИ ЧЕРЕЗ UNSLOTH
# =============================================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["model_name"],
    max_seq_length=cfg["max_seq_length"],
    dtype=torch.bfloat16,
    load_in_4bit=cfg["load_in_4bit"],
)

print("\nМодель загружена")
print(f"   Всего параметров: {model.num_parameters():,}")

if torch.cuda.is_available():
    used_mb = torch.cuda.memory_allocated() / 1e6
    print(f"   VRAM после загрузки: {used_mb:.0f} MB")

# =============================================================================
# 3. ДОБАВЛЕНИЕ LORA-АДАПТЕРОВ
# =============================================================================

model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_r"],
    # Ранг LoRA-матриц. Для каждого target_module создаётся две матрицы:
    # A размером (hidden_dim × r) и B размером (r × hidden_dim).
    # При r=16 и hidden_dim=1536: A = 1536×16 = 24K параметров,
    # B = 16×1536 = 24K параметров на каждый модуль.
    lora_alpha=cfg["lora_alpha"],
    # Масштабирующий коэффициент. Итоговое обновление веса считается как:
    # W_new = W_frozen + (lora_alpha / r) × B × A
    # При alpha=32, r=16: scaling = 2.0. Это множитель эффективного lr.
    lora_dropout=cfg["lora_dropout"],
    target_modules=cfg["lora_target_modules"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    # Ключевая оптимизация VRAM при обучении.
    # Стандартный PyTorch хранит все промежуточные активации для backward pass.
    # Gradient checkpointing: активации НЕ хранятся, а перевычисляются при
    # backward — это в ~3x меньше VRAM ценой ~20% замедления.
    # Значение "unsloth" (вместо True) использует кастомную реализацию Unsloth,
    # которая умнее выбирает что кэшировать — экономит ещё больше VRAM.
    random_state=42,
)

trainable = model.num_parameters(only_trainable=True)
total = model.num_parameters()
print("\nLoRA-адаптеры добавлены")
print(
    f"   Обучаемых параметров: {trainable:,}  ({100 * trainable / total:.2f}% от всей модели)"
)
print(f"   Замороженных:         {total - trainable:,}")

# =============================================================================
# 4. ЗАГРУЗКА И ПОДГОТОВКА ДАТАСЕТОВ
# =============================================================================


def format_sample(sample: dict) -> dict:
    """
    Применяет chat template к одному примеру датасета.

    Вход:  {"messages": [{"role": "user", "content": "..."}, ...], "source": "..."}
    Выход: {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n..."}

    Почему не передаём messages напрямую в SFTTrainer:
    SFTTrainer умеет работать с messages-форматом, но тогда применение
    chat template происходит внутри тренера с дефолтными настройками.
    Делая это явно, мы контролируем точно тот же формат, что и при инференсе.
    """
    return {
        "text": tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
            # False = не добавлять токен начала ответа ("<|im_start|>assistant\n")
            # в конец. При обучении нам нужна полная последовательность
            # включая ответ ассистента с закрывающим <|im_end|>.
            # add_generation_prompt=True нужен только при инференсе —
            # чтобы модель знала, что дальше должна идти её генерация.
        )
    }


def is_within_length(sample: dict) -> bool:
    """
    Фильтр: пропускаем примеры длиннее max_seq_length.
    Возвращает True если пример укладывается в лимит, False — если нет.
    """
    text = tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tokens <= cfg["max_seq_length"]


# Загружаем датасеты из jsonl-файлов
train_dataset = load_dataset(
    "json",
    data_files="/home/danil-pc/fine-tuning_project/data/final/train.jsonl",
    split="train",
)

val_dataset = load_dataset(
    "json",
    data_files="/home/danil-pc/fine-tuning_project/data/final/val.jsonl",
    split="train",
)

# Фильтруем выброс (пример с 1056 токенами)
train_before = len(train_dataset)
train_dataset = train_dataset.filter(is_within_length)
# filter() проходит по всем примерам и оставляет только те,
# для которых функция вернула True. Работает батчами, эффективно.
filtered = train_before - len(train_dataset)
print("\nДатасеты загружены")
print(f"   Train: {len(train_dataset)} примеров (отфильтровано: {filtered})")
print(f"   Val:   {len(val_dataset)} примеров")

# Применяем chat template ко всем примерам
train_dataset = train_dataset.map(format_sample)
val_dataset = val_dataset.map(format_sample)

# =============================================================================
# 5. КАСТОМНЫЙ MLFLOW CALLBACK
# =============================================================================


class MLflowCallback(TrainerCallback):
    """
    Callback для логирования метрик в MLflow.

    Trainer вызывает on_log каждые logging_steps шагов.
    Мы перехватываем это событие и пишем метрики в текущий MLflow run.

    Почему не используем встроенный MLflow в Trainer (report_to="mlflow"):
    Встроенная интеграция логирует метрики с именами без префиксов
    (просто "loss" вместо "train/loss") и не позволяет кастомизировать
    что именно логировать. Наш callback даёт полный контроль.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Вызывается Trainer'ом каждые logging_steps шагов.

        args    — объект SFTConfig с настройками обучения
        state   — текущее состояние: шаг, эпоха, лучшая метрика
        control — объект для управления обучением (можно остановить)
        logs    — словарь с метриками текущего шага
        """
        if not logs:
            return
        # logs может быть None если вызов произошёл без новых данных

        step = state.global_step
        # global_step — суммарное количество выполненных optimizer steps.
        # При gradient_accumulation_steps=4 это НЕ количество forward pass'ов,
        # а именно шагов оптимизатора (каждые 4 forward pass = 1 global_step).

        metrics = {}

        if "loss" in logs:
            metrics["train/loss"] = logs["loss"]

        if "eval_loss" in logs:
            metrics["eval/loss"] = logs["eval_loss"]

        if "learning_rate" in logs:
            metrics["train/learning_rate"] = logs["learning_rate"]
            # Текущий lr после планировщика (cosine decay + warmup).
            # Полезно видеть на графике что lr ведёт себя как задумано.

        if "grad_norm" in logs:
            metrics["train/grad_norm"] = logs["grad_norm"]
            # L2-норма градиентов перед gradient clipping.
            # Если grad_norm систематически растёт или взрывается (>10) —
            # сигнал проблемы. В норме: нестабильно в начале, стабилизируется.

        if torch.cuda.is_available():
            metrics["system/gpu_memory_allocated_gb"] = (
                torch.cuda.memory_allocated() / 1e9
            )

            metrics["system/gpu_memory_reserved_gb"] = (
                torch.cuda.memory_reserved() / 1e9
            )

        if metrics:
            mlflow.log_metrics(metrics, step=step)


# =============================================================================
# 6. КОНФИГУРАЦИЯ ТРЕНЕРА
# =============================================================================

training_args = SFTConfig(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["num_train_epochs"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    # Размер батча на одно устройство. Если бы была мультигпу-система,
    # effective batch = per_device × num_gpus × grad_accum. У нас 1 GPU.
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    warmup_steps=cfg.get("warmup_steps", 10),
    # Доля от общего числа шагов для linear warmup.
    # Trainer сам вычислит: warmup_steps = total_steps × warmup_ratio.
    learning_rate=cfg["learning_rate"],
    # Пиковый lr — достигается после warmup, потом cosine decay.
    lr_scheduler_type=cfg["lr_scheduler_type"],
    # Тип планировщика lr. "cosine" = cosine annealing без перезапусков.
    weight_decay=cfg["weight_decay"],
    # L2-регуляризация. Trainer передаёт это в оптимизатор.
    # AdamW применяет weight decay к весам, но НЕ к bias и LayerNorm.
    optim=cfg["optim"],
    # Имя оптимизатора. Trainer создаёт объект оптимизатора по этой строке.
    # "paged_adamw_8bit" требует установленного bitsandbytes.
    bf16=cfg["bf16"],
    fp16=cfg["fp16"],
    # Эти флаги говорят Trainer'у использовать mixed precision.
    # Trainer обернёт forward и backward pass в autocast context.
    logging_steps=cfg["logging_steps"],
    eval_strategy="steps",
    eval_steps=cfg["eval_steps"],
    save_strategy="steps",
    save_steps=cfg["save_steps"],
    save_total_limit=cfg["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataset_text_field="text",
    max_length=cfg["max_seq_length"],
    # SFTTrainer обрежет последовательности длиннее этого значения.
    # У нас уже отфильтрован единственный выброс, так что обрезки не будет —
    # но параметр всё равно нужен для правильного паддинга батчей.
    packing=False,
    # Packing = "упаковка" коротких примеров в одну длинную последовательность
    # чтобы не было пустого паддинга. Теоретически ускоряет обучение,
    # но на практике с нашими данными (средняя длина ~200-400 токенов,
    # max_seq_length=560) выгода минимальна, а риск артефактов — есть.
    # Оставляем False для простоты и предсказуемости.
    eos_token="<|im_end|>",
)
# =============================================================================
# 7. СБОРКА ТРЕНЕРА
# =============================================================================

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    callbacks=[MLflowCallback()],
    # Список callbacks. Trainer будет вызывать их в нужные моменты.
    # Наш MLflowCallback добавляется к дефолтным (ProgressCallback, PrinterCallback).
)

# =============================================================================
# 8. ЗАПУСК ОБУЧЕНИЯ
# =============================================================================

mlflow.set_experiment(cfg["mlflow_experiment_name"])

with mlflow.start_run(run_name=cfg["mlflow_run_name"]) as run:
    # Логируем все гиперпараметры одним вызовом
    mlflow.log_params(
        {
            "model_name": cfg["model_name"],
            "max_seq_length": cfg["max_seq_length"],
            "lora_r": cfg["lora_r"],
            "lora_alpha": cfg["lora_alpha"],
            "lora_dropout": cfg["lora_dropout"],
            "learning_rate": cfg["learning_rate"],
            "num_train_epochs": cfg["num_train_epochs"],
            "batch_size": cfg["per_device_train_batch_size"],
            "grad_accumulation": cfg["gradient_accumulation_steps"],
            "effective_batch_size": cfg["per_device_train_batch_size"]
            * cfg["gradient_accumulation_steps"],
            "optimizer": cfg["optim"],
            "lr_scheduler": cfg["lr_scheduler_type"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "trainable_params": model.num_parameters(only_trainable=True),
        }
    )

    print(f"\nMLflow Run ID: {run.info.run_id}")
    print(f"Experiment:    {cfg['mlflow_experiment_name']}")
    print("\nНачинаем обучение...")
    print(
        f"Шагов итого: ~{len(train_dataset) // cfg['per_device_train_batch_size'] * cfg['num_train_epochs']}"
    )

    # ── Старт ──────────────────────────────────────────────────────────────
    trainer_stats = trainer.train()

    # ── Финальные метрики ───────────────────────────────────────────────────
    mlflow.log_metrics(
        {
            "final/train_loss": trainer_stats.training_loss,
            "final/train_runtime_sec": trainer_stats.metrics["train_runtime"],
            "final/samples_per_second": trainer_stats.metrics[
                "train_samples_per_second"
            ],
        }
    )

    # ── Сохранение адаптера ─────────────────────────────────────────────────
    adapter_path = cfg["output_dir"] + "/final_adapter"

    model.save_pretrained(adapter_path)

    tokenizer.save_pretrained(adapter_path)

    # Логируем адаптер и конфиг как артефакты MLflow
    mlflow.log_artifact(adapter_path, artifact_path="lora_adapter")

    mlflow.log_artifact(
        str(_PROJECT_ROOT / "configs" / "training_config.yaml"),
        artifact_path="config",
    )

    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    print(f"\nОбучение завершено за {runtime_min:.1f} минут")
    print(f"   Финальный train loss: {trainer_stats.training_loss:.4f}")
    print(f"   Адаптер сохранён:     {adapter_path}")
    print(f"   MLflow run ID:        {run.info.run_id}")
    print("\nЗапусти MLflow UI:  mlflow ui --backend-store-uri mlflow/mlruns")
