"""
Повторный прогон LLM-as-a-judge для coding_eval результатов где судья не отработал.
Берёт готовые коды из артефакта, не гоняет модель заново.

Запуск:
  python scripts/rerun_coding_judge.py \
      --artifact mlflow/artifacts/baseline_before_training/coding.json \
      --run_name baseline_before_training
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from eval.coding_eval import _call_coding_judge

import mlflow

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow/mlflow.db")


def rerun_judge(artifact_path: str, run_name: str):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    with open(artifact_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    n_fixed = 0

    print(f"Loaded {len(results)} results from {artifact_path}")
    print(
        f"Results with judge errors: {sum(1 for r in results if r['quality_score'] == -1)}\n"
    )

    for result in results:
        if result["quality_score"] != -1:
            print(
                f"  [{result['id']}] Already has score {result['quality_score']}/5 — skipping"
            )
            continue

        print(f"  [{result['id']}] Rerunning judge...")

        verdict = _call_coding_judge(
            api_key=api_key,
            task_prompt=result["prompt"],
            code=result["generated_code"],
            passed_tests=result["passed"],
        )

        result["quality_score"] = verdict.get("quality_score", -1)
        result["is_pythonic"] = verdict.get("is_pythonic", False)
        result["handles_edge_cases"] = verdict.get("handles_edge_cases", False)
        result["complexity_note"] = verdict.get("complexity_note", "")
        result["review"] = verdict.get("review", "")

        if result["quality_score"] != -1:
            n_fixed += 1
            print(
                f"    Score: {result['quality_score']}/5 | {result['review'][:80]}..."
            )
        else:
            print(f"    Judge failed again: {result['review']}")

    # Пересчитываем итоговые метрики
    valid_quality = [
        r["quality_score"] for r in results if r["quality_score"] > 0
    ]
    avg_quality = (
        round(sum(valid_quality) / len(valid_quality), 3)
        if valid_quality
        else 0.0
    )

    data["avg_quality_score"] = avg_quality
    data["results"] = results

    # Сохраняем обновлённый артефакт
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nFixed {n_fixed} results")
    print(f"Updated avg_quality_score: {avg_quality:.3f}/5.0")
    print(f"Saved to: {artifact_path}")

    # Обновляем метрику в MLflow
    mlflow.set_experiment("qwen-coder-finetuning")
    runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    if not runs.empty:
        run_id = runs.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("coding_avg_quality", avg_quality)
            mlflow.log_artifact(artifact_path)
        print(f"MLflow run '{run_name}' updated with new avg_quality_score")
    else:
        print(
            f"MLflow run '{run_name}' not found — metrics not updated in MLflow"
        )
        print(f"   Updated artifact saved locally: {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=str,
        default="mlflow/artifacts/baseline_before_training/coding.json",
        help="Path to coding.json artifact",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline_before_training",
        help="MLflow run name to update",
    )
    args = parser.parse_args()
    rerun_judge(args.artifact, args.run_name)
