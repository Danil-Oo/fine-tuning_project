"""
04_evaluate.py — Базовая оценка модели (до и после обучения).

Запуск:
  # Baseline (до обучения):
  python scripts/04_evaluate.py --run_name baseline_before_training

  # После обучения (с LoRA адаптерами):
  python scripts/04_evaluate.py --run_name after_finetuning --adapter_path adapters/qwen-qlora

  # Только отдельные компоненты:
  python scripts/04_evaluate.py --run_name test --skip_domain --skip_humaneval

Переменная окружения:
  OPENROUTER_API_KEY=<your_key>
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import mlflow

# Явно указываем SQLite как бэкенд
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB = f"sqlite:///{PROJECT_ROOT}/mlflow/mlflow.db"
MLFLOW_ARTIFACTS = f"{PROJECT_ROOT}/mlflow/artifacts"

mlflow.set_tracking_uri(MLFLOW_DB)
# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from eval.coding_eval import run_coding_evaluation
from eval.domain_eval import run_domain_evaluation
from eval.humaneval_mini import run_humaneval_mini
from eval.perplexity import compute_perplexity, load_val_data

# ─── Конфигурация ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
VAL_FILE = "data/final/val.jsonl"
DOMAIN_QUES = "data/eval/domain_questions.json"
CODING_TASKS = "data/eval/coding_tasks.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Загрузка модели ─────────────────────────────────────────────────────────


def load_model_4bit(model_path: str):
    """
    Загружает модель в 4-bit NF4 с двойной квантизацией.
    Именно в таком формате модель будет обучаться и работать.
    """
    log.info(f"Loading model from: {model_path}")
    log.info("Quantization: 4-bit NF4 + double quantization")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # double quantization: экономит ~0.4 бит/параметр
        bnb_4bit_compute_dtype=torch.float16,  # вычисления в fp16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = next(model.parameters()).device
    log.info(f"Model loaded on device: {device}")

    # Выводим примерный размер модели в памяти
    try:
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        log.info(f"GPU memory used after loading: {mem_mb:.0f} MB")
    except Exception:
        pass

    return model, tokenizer, str(device)


def load_model_with_adapter(base_model_path: str, adapter_path: str):
    """Загружает базовую модель + LoRA адаптеры для оценки после обучения."""

    model, tokenizer, device = load_model_4bit(base_model_path)
    log.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    log.info("Adapter loaded successfully")
    return model, tokenizer, device


# ─── Утилиты ─────────────────────────────────────────────────────────────────


def save_artifacts(results: dict, run_name: str) -> list[str]:
    """Сохраняет детальные результаты как JSON артефакты для MLflow."""
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    artifacts_dir = PROJECT_ROOT / "mlflow" / "artifacts" / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, data in results.items():
        path = artifacts_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        saved.append(str(path))
        log.info(f"Saved artifact: {path}")

    return saved


def print_summary(results: dict):
    """Выводит итоговую сводку в консоль."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if "perplexity" in results:
        p = results["perplexity"]
        print("\nPerplexity")
        print(f"   Overall:    {p['perplexity']:.4f}")
        print(f"   Avg loss:   {p['avg_loss']:.6f}")
        print(f"   Samples:    {p['n_samples']}")
        if p.get("perplexity_by_category"):
            for cat, ppl in p["perplexity_by_category"].items():
                print(f"   [{cat}]:  {ppl:.4f}")

    if "domain" in results:
        d = results["domain"]
        print("\nDomain Eval (LLM Architecture)")
        print(f"   Avg score:  {d['avg_score']:.3f} / 5.0")
        print(f"   Evaluated:  {d['n_evaluated']} questions")
        if d.get("scores_by_category"):
            for cat, score in d["scores_by_category"].items():
                print(f"   [{cat}]:  {score:.3f}/5.0")

    if "coding" in results:
        c = results["coding"]
        print("\nCoding Eval")
        print(
            f"   Pass rate:  {c['pass_rate']:.1%} ({int(c['pass_rate'] * c['n_tasks'])}/{c['n_tasks']})"
        )
        print(f"   Avg quality: {c['avg_quality_score']:.2f} / 5.0")
        if c.get("pass_rate_by_difficulty"):
            for diff, rate in c["pass_rate_by_difficulty"].items():
                print(f"   [{diff}]:  {rate:.1%}")

    if "humaneval" in results:
        h = results["humaneval"]
        print("\nHumanEval Mini (pass@1)")
        print(
            f"   pass@1:     {h['pass_at_1']:.4f} ({h['n_passed']}/{h['n_tasks']})"
        )

    print("\n" + "=" * 60)


# ─── Главная функция ──────────────────────────────────────────────────────────


def evaluate(
    run_name: str,
    adapter_path: str = None,
    skip_perplexity: bool = False,
    skip_domain: bool = False,
    skip_coding: bool = False,
    skip_humaneval: bool = False,
    n_domain_questions: int = None,
    n_coding_tasks: int = None,
):
    # api_key = os.environ.get("OPENROUTER_API_KEY")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and not (skip_domain and skip_coding):
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set.\n"
            "Set it with: export OPENROUTER_API_KEY=your_key\n"
            "Or skip LLM-judge evals: --skip_domain --skip_coding"
        )

    # Загружаем модель
    if adapter_path:
        model, tokenizer, device = load_model_with_adapter(
            MODEL_NAME, adapter_path
        )
    else:
        model, tokenizer, device = load_model_4bit(MODEL_NAME)

    all_results = {}

    # ── 1. Perplexity ────────────────────────────────────────────────────────
    if not skip_perplexity:
        log.info("\n[1/4] Computing perplexity on val set...")
        val_data = load_val_data(VAL_FILE)
        ppl_results = compute_perplexity(model, tokenizer, val_data, device)
        all_results["perplexity"] = ppl_results
        log.info(f"Perplexity: {ppl_results['perplexity']:.4f}")
    else:
        log.info("[1/4] Skipping perplexity")

    # ── 2. Domain evaluation ─────────────────────────────────────────────────
    if not skip_domain:
        log.info("\n[2/4] Running domain evaluation (LLM-as-a-judge)...")
        domain_results = run_domain_evaluation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            questions_path=DOMAIN_QUES,
            api_key=api_key,
            n_questions=n_domain_questions,
        )
        all_results["domain"] = domain_results
    else:
        log.info("[2/4] Skipping domain evaluation")

    # ── 3. Coding evaluation ─────────────────────────────────────────────────
    if not skip_coding:
        log.info(
            "\n[3/4] Running coding evaluation (unit tests + LLM-as-a-judge)..."
        )
        coding_results = run_coding_evaluation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            tasks_path=CODING_TASKS,
            api_key=api_key,
            n_tasks=n_coding_tasks,
        )
        all_results["coding"] = coding_results
    else:
        log.info("[3/4] Skipping coding evaluation")

    # ── 4. HumanEval Mini ────────────────────────────────────────────────────
    if not skip_humaneval:
        log.info("\n[4/4] Running HumanEval Mini (pass@1)...")
        humaneval_results = run_humaneval_mini(model, tokenizer, device)
        all_results["humaneval"] = humaneval_results
    else:
        log.info("[4/4] Skipping HumanEval")

    # ── Логируем в MLflow ────────────────────────────────────────────────────
    # mlflow.set_experiment("qwen-coder-finetuning")
    experiment_name = "qwen-coder-finetuning"
    artifact_location = f"file:///{PROJECT_ROOT}/mlflow/artifacts"

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Параметры
        mlflow.log_param(
            "model_path", adapter_path if adapter_path else MODEL_NAME
        )
        mlflow.log_param("quantization", "4bit_nf4_double_quant")
        mlflow.log_param("timestamp", datetime.now().isoformat())
        mlflow.log_param("is_finetuned", adapter_path is not None)

        # Метрики
        if "perplexity" in all_results:
            p = all_results["perplexity"]
            mlflow.log_metric("perplexity", p["perplexity"])
            mlflow.log_metric("avg_loss", p["avg_loss"])
            mlflow.log_metric("perplexity_n_samples", p["n_samples"])
            for cat, ppl in p.get("perplexity_by_category", {}).items():
                mlflow.log_metric(f"perplexity_{cat}", ppl)

        if "domain" in all_results:
            d = all_results["domain"]
            mlflow.log_metric("domain_avg_score", d["avg_score"])
            mlflow.log_metric("domain_n_evaluated", d["n_evaluated"])
            for cat, score in d.get("scores_by_category", {}).items():
                mlflow.log_metric(f"domain_score_{cat}", score)

        if "coding" in all_results:
            c = all_results["coding"]
            mlflow.log_metric("coding_pass_rate", c["pass_rate"])
            mlflow.log_metric("coding_avg_quality", c["avg_quality_score"])
            mlflow.log_metric("coding_n_tasks", c["n_tasks"])
            for diff, rate in c.get("pass_rate_by_difficulty", {}).items():
                mlflow.log_metric(f"coding_pass_{diff}", rate)

        if "humaneval" in all_results:
            h = all_results["humaneval"]
            mlflow.log_metric("humaneval_pass_at_1", h["pass_at_1"])
            mlflow.log_metric("humaneval_n_passed", h["n_passed"])
            mlflow.log_metric("humaneval_n_tasks", h["n_tasks"])

        # Артефакты (детальные результаты)
        artifact_paths = save_artifacts(all_results, run_name)
        for path in artifact_paths:
            mlflow.log_artifact(path)

        log.info(f"\nResults logged to MLflow run: '{run_name}'")

    print_summary(all_results)
    return all_results


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Coder-1.5B-Instruct before and after fine-tuning"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline_before_training",
        help="MLflow run name (e.g. 'baseline_before_training' or 'after_finetuning')",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapters (only for post-training evaluation)",
    )
    parser.add_argument("--skip_perplexity", action="store_true")
    parser.add_argument("--skip_domain", action="store_true")
    parser.add_argument("--skip_coding", action="store_true")
    parser.add_argument("--skip_humaneval", action="store_true")
    parser.add_argument(
        "--n_domain_questions",
        type=int,
        default=None,
        help="Limit number of domain questions (default: all 20)",
    )
    parser.add_argument(
        "--n_coding_tasks",
        type=int,
        default=None,
        help="Limit number of coding tasks (default: all 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        run_name=args.run_name,
        adapter_path=args.adapter_path,
        skip_perplexity=args.skip_perplexity,
        skip_domain=args.skip_domain,
        skip_coding=args.skip_coding,
        skip_humaneval=args.skip_humaneval,
        n_domain_questions=args.n_domain_questions,
        n_coding_tasks=args.n_coding_tasks,
    )
