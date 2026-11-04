"""Extract answer labels and evaluate predictions against JSONL ground truth."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any


LABEL_SET = {"up", "down", "changed", "unchanged"}
OTHER_LABEL_PATTERNS = (
    r"Answer:\s*(up|down|changed|unchanged)$",
    r"(up|down|changed|unchanged)$",
    r"(up|down|changed|unchanged)\.$",
    r'"(up|down|changed|unchanged)\."$',
    r'"(up|down|changed|unchanged)"$',
)


def build_parser() -> argparse.ArgumentParser:
    """Declare all filesystem inputs and outputs in one place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data_samples",
        help="Directory of ground-truth JSONL files (default: bundled data_samples).",
    )
    parser.add_argument("--input-file", type=Path,
                        help="Evaluate this ground-truth file instead of input-dir.")
    predictions = parser.add_mutually_exclusive_group()
    predictions.add_argument("--prediction-dir", type=Path,
                             help="Directory of prediction files with matching input filenames.")
    predictions.add_argument("--prediction-file", type=Path,
                             help="Prediction file paired with an explicit input-file.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Write one answer_metrics.json file per input; otherwise print JSON.",
    )
    parser.add_argument("--pattern", default="*.jsonl", help="Input filename glob.")
    parser.add_argument(
        "--strict", choices=["v1"], default=None,
        help="Also accept an Answer: label line anywhere in a model output.",
    )
    return parser


def _class_sort_key(label: str) -> int:
    return {"changed": 0, "unchanged": 1, "up": 2, "down": 3}.get(label, 100)


def parse_label_from_text(
    text: str, label_set: set[str] | None = None, *, strict: str | None = None,
) -> str | None:
    """Extract an answer using the original parser's precedence and fallbacks.

    Boxed and answer-tag values are returned without label-set validation, as in
    the original evaluator. The optional strict mode adds a line-based fallback.
    """
    if not text:
        return None
    model_output = text.strip()
    labels = label_set if label_set is not None else LABEL_SET

    match_boxed = re.search(r"\s*\\boxed\{(.*?)\}", model_output)
    if match_boxed:
        return match_boxed.group(1).strip().lower()
    match_answer_tag = re.search(r"<answer>(.*?)</answer>", model_output, re.IGNORECASE)
    if match_answer_tag:
        return match_answer_tag.group(1).strip().lower()
    if model_output in labels:
        return model_output.lower()
    first_line = model_output.split("\n", 1)[0]
    if first_line in labels:
        return first_line.lower()
    for pattern in OTHER_LABEL_PATTERNS:
        match = re.search(pattern, model_output.replace("*", ""), re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    if strict == "v1":
        match_answer_line = re.search(
            r"^\s*Answer:\s*(up|down|changed|unchanged)\s*$", model_output,
            re.IGNORECASE | re.MULTILINE,
        )
        if match_answer_line and match_answer_line.group(1).lower() in labels:
            return match_answer_line.group(1).lower()
    return None


def _compute_scores(y_true: list, y_pred: list) -> dict[str, Any]:
    """Compute the original sklearn metrics without pooling separate files."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, precision_score,
        recall_score,
    )

    if len(y_true) != len(y_pred):
        raise ValueError("Ground-truth and prediction counts must match.")
    scores: dict[str, Any] = {
        "accuracy": 0.0, "balanced_accuracy": 0.0,
        "f1_micro": 0.0, "f1_weighted": 0.0,
        "num_samples": len(y_true), "major_class_ratio": 0.0,
    }
    if not y_true:
        return scores
    scores["accuracy"] = float(accuracy_score(y_true, y_pred))
    scores["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    for average in ("micro", "weighted"):
        scores[f"f1_{average}"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        )
    scores["major_class_ratio"] = max(Counter(y_true).values()) / len(y_true)
    labels = sorted(set(y_true), key=_class_sort_key)
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for label, precision, recall, f1 in zip(labels, precisions, recalls, f1s):
        scores[f"precision_{label}"] = float(precision)
        scores[f"recall_{label}"] = float(recall)
        scores[f"f1_{label}"] = float(f1)
    return scores


def match_records(
    ground_truths: list[dict[str, Any]], predictions: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """Match IDs while preserving the original duplicate and unmatched-ID rules.

    The last ground-truth record and first prediction win for duplicate IDs.
    Every prediction must provide an ID; missing ID keys raise an error.
    Unknown IDs are ignored; missing predictions do not enter the score.
    Inputs are not modified.
    """
    ground_truth_by_id = {record["id"]: record for record in ground_truths}
    candidates = [dict(record) for record in predictions]
    for index, prediction in enumerate(candidates, 1):
        if "id" not in prediction:
            raise ValueError(f"Prediction record {index} is missing 'id'; every prediction must provide an ID.")
    seen: set[Any] = set()
    unique_predictions = []
    for prediction in candidates:
        if prediction["id"] not in seen:
            seen.add(prediction["id"])
            unique_predictions.append(prediction)
    matched = [
        (ground_truth_by_id[prediction["id"]], prediction)
        for prediction in unique_predictions if prediction["id"] in ground_truth_by_id
    ]
    counts = {
        "*num_gt": len(ground_truths),
        "*num_gen": len(matched),
        "*num_gen_filtered_out_not_in_gt": len(unique_predictions) - len(matched),
        "*num_remaining": len(ground_truths) - len(matched),
    }
    return matched, counts


def _ground_truth_label(record: dict[str, Any]) -> str:
    metadata = record.get("input") or {}
    label = metadata["label"] if "label" in metadata else record["label"]
    return label.strip().lower()


def evaluate_records(
    ground_truths: list[dict[str, Any]], predictions: list[dict[str, Any]],
    *, strict: str | None = None,
) -> dict[str, Any]:
    """Evaluate one dataset's answers and report matching and parsing counts."""
    matched, counts = match_records(ground_truths, predictions)
    if not matched:
        return {"_no_answer_in_generation": 0, **counts}
    y_true, y_pred = [], []
    no_answer = 0
    for ground_truth, prediction in matched:
        label = parse_label_from_text(prediction.get("model_output", "").strip(), strict=strict)
        y_true.append(_ground_truth_label(ground_truth))
        if label is None:
            label = "__invalid_prediction__"
            no_answer += 1
        y_pred.append(label)
    scores = _compute_scores(y_true, y_pred)
    scores["_no_answer_in_generation"] = no_answer
    for label, count in sorted(Counter(y_true).items(), key=lambda item: _class_sort_key(item[0])):
        scores[f"_num_{label}"] = count
    scores.update(counts)
    return scores


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL lines, with line numbers in parsing errors."""
    records = []
    with Path(path).open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            records.append(record)
    return records


def evaluate_directory(
    input_dir: str | Path, prediction_dir: str | Path | None = None,
    *, pattern: str = "*.jsonl", strict: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate matching filenames independently; never aggregate file scores."""
    input_dir = Path(input_dir)
    use_sample_pair = prediction_dir is None
    prediction_dir = Path(prediction_dir) if prediction_dir is not None else input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_dir}")
    sample_input = input_dir / "samples_input.jsonl"
    sample_output = input_dir / "samples_output.jsonl"
    paths = sorted(path for path in input_dir.glob(pattern)
                   if not (sample_input.is_file() and path == sample_output))
    if not paths:
        raise FileNotFoundError(f"No input files match {pattern!r} in {input_dir}.")
    results = {}
    for ground_truth_path in paths:
        prediction_path = prediction_dir / ground_truth_path.name
        if use_sample_pair and ground_truth_path == sample_input and sample_output.is_file():
            prediction_path = sample_output
        scores = evaluate_records(
            load_jsonl(ground_truth_path), load_jsonl(prediction_path), strict=strict,
        )
        if scores["*num_gen"] == 0:
            print(
                f"Warning: No matching ground-truth IDs in {ground_truth_path.name}; "
                "skipping metric computation.", file=sys.stderr,
            )
            continue
        scores["_gt_file_name"] = ground_truth_path.name
        results[ground_truth_path.name] = scores
    return results




def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prediction_file is not None and args.input_file is None:
        parser.error("--prediction-file requires --input-file.")
    input_file, prediction_file = args.input_file, args.prediction_file
    try:
        if input_file is not None:
            if prediction_file is None:
                prediction_file = (args.prediction_dir / input_file.name
                                   if args.prediction_dir is not None else input_file)
            scores = evaluate_records(load_jsonl(input_file), load_jsonl(prediction_file), strict=args.strict)
            results = {}
            if scores["*num_gen"]:
                scores["_gt_file_name"] = input_file.name
                results[input_file.name] = scores
            else:
                print(f"Warning: No matching ground-truth IDs in {input_file.name}; "
                      "skipping metric computation.", file=sys.stderr)
        else:
            results = evaluate_directory(args.input_dir, args.prediction_dir,
                                         pattern=args.pattern, strict=args.strict)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        parser.error(str(error))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, scores in results.items():
        payload = json.dumps(scores, indent=2)
        if args.output_dir is None:
            print(json.dumps({"file": filename, "metrics": scores}, indent=2))
        else:
            output = args.output_dir / f"{Path(filename).stem}.answer_metrics.json"
            output.write_text(payload + "\n", encoding="utf-8")
            print(f"Saved {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
