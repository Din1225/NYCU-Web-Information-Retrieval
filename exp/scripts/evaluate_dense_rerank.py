from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense/rerank results with masked annotations.")
    parser.add_argument("--annotation", required=True, help="Path to annotation JSON.")
    parser.add_argument("--result", required=True, help="Path to experiment result JSON.")
    parser.add_argument(
        "--result_field",
        default="both",
        choices=["both", "dense_results", "rerank_results"],
        help="Which result field to evaluate.",
    )
    parser.add_argument("--output", default=None, help="Optional evaluation output JSON path.")
    parser.add_argument(
        "--binary_relevance_threshold",
        type=int,
        default=2,
        choices=[1, 2],
        help="Label threshold used by MAP/MRR. Relevant means label >= threshold.",
    )
    return parser.parse_args()


def resolve_existing_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    exp_candidate = EXP_ROOT / candidate
    if exp_candidate.exists():
        return exp_candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    return exp_candidate


def resolve_output_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return EXP_ROOT / candidate


def default_output_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.evaluation.json")


def main() -> None:
    args = parse_args()

    from src.evaluation import (
        MaskedEvaluationConfig,
        evaluate_multiple_result_fields,
        evaluate_result_file,
        save_evaluation,
    )

    annotation_path = resolve_existing_path(args.annotation)
    result_path = resolve_existing_path(args.result)
    output_path = resolve_output_path(args.output) if args.output else default_output_path(result_path)

    if args.result_field == "both":
        evaluation = evaluate_multiple_result_fields(
            annotation_path=annotation_path,
            result_path=result_path,
            result_fields=["dense_results", "rerank_results"],
            binary_relevance_threshold=args.binary_relevance_threshold,
            top_ks=(10, 30),
        )
    else:
        evaluation = evaluate_result_file(
            MaskedEvaluationConfig(
                annotation_path=annotation_path,
                result_path=result_path,
                result_field=args.result_field,
                binary_relevance_threshold=args.binary_relevance_threshold,
                top_ks=(10, 30),
            )
        )

    save_evaluation(evaluation, output_path)
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
