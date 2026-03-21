#!/usr/bin/env python3
"""
Batch MNIST inference for openfhe-numpy.

Runs both LoLa and LeNet-5 models over a configurable range of MNIST test
samples, collecting plaintext predictions, ciphertext predictions, and true
labels.  Prints summary statistics and writes per-sample results to CSV.

Usage examples:
    # First 10 samples, scheme switching, both models
    python scripts/batch_inference.py --start 0 --end 10

    # Samples 500-509, chebyshev activation (degree 119)
    python scripts/batch_inference.py --start 500 --end 510 --activation cheby --cheby-degree 119

    # Only lola model, samples 0-99, std128 security
    python scripts/batch_inference.py --end 100 --model lola --security std128

    # Custom output directory
    python scripts/batch_inference.py --end 50 --output-dir results/run1
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
EXECUTABLES = {
    "lola": BUILD_DIR / "mnist-lola",
    "lenet5": BUILD_DIR / "mnist-lenet5",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch MNIST inference over openfhe-numpy models."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start sample index (inclusive, default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=10, help="End sample index (exclusive, default: 10)"
    )
    parser.add_argument(
        "--model",
        choices=["lola", "lenet5", "both"],
        default="both",
        help="Which model to run (default: both)",
    )
    parser.add_argument(
        "--activation",
        choices=["scheme", "cheby"],
        default="scheme",
        help="Activation type (default: scheme)",
    )
    parser.add_argument(
        "--security",
        choices=["toy", "std128"],
        default="toy",
        help="Security level for scheme switching (default: toy)",
    )
    parser.add_argument(
        "--cheby-degree",
        type=int,
        default=119,
        help="Chebyshev polynomial degree (default: 119)",
    )
    parser.add_argument(
        "--optimize",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable optimizations (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CSV output (default: results/<timestamp>)",
    )
    return parser.parse_args()


def build_cmd(model: str, sample_idx: int, args) -> list[str]:
    """Build the command-line invocation for a single inference."""
    exe = str(EXECUTABLES[model])
    cmd = [exe, str(sample_idx), args.activation]

    if args.activation == "scheme":
        cmd.append(args.security)
        cmd.append(str(args.optimize))
    elif args.activation == "cheby":
        cmd.append(str(args.cheby_degree))
        cmd.append(str(args.optimize))

    return cmd


# Patterns to extract from stdout
RE_CLEARTEXT_PRED = re.compile(r"Cleartext predicted class:\s*(\d+)")
RE_ENCRYPTED_PRED = re.compile(r"Predicted class:\s*(\d+)")
RE_TRUE_LABEL = re.compile(r"True label:\s*(\d+)")
RE_INFERENCE_TIME = re.compile(r"Total inference time:\s*([\d.e+\-]+)\s*ms")
RE_LOADED_LABEL = re.compile(r"Loaded sample with true label:\s*(\d+)")
RE_SAMPLE_NOT_FOUND = re.compile(r"Could not find MNIST sample")
RE_FINAL_MAX_ERROR = re.compile(r"Final output max error:\s*([\d.e+\-]+)")
RE_FINAL_AVG_ERROR = re.compile(r"Final output avg error:\s*([\d.e+\-]+)")


def parse_output(stdout: str) -> dict:
    """Parse C++ executable stdout into a result dict."""
    result = {
        "cleartext_pred": None,
        "encrypted_pred": None,
        "true_label": None,
        "inference_time_ms": None,
        "final_max_error": None,
        "final_avg_error": None,
        "error": None,
    }

    if RE_SAMPLE_NOT_FOUND.search(stdout):
        result["error"] = "sample_not_found"
        return result

    m = RE_CLEARTEXT_PRED.search(stdout)
    if m:
        result["cleartext_pred"] = int(m.group(1))

    m = RE_ENCRYPTED_PRED.search(stdout)
    if m:
        result["encrypted_pred"] = int(m.group(1))

    # Try both patterns for true label
    m = RE_TRUE_LABEL.search(stdout)
    if m:
        result["true_label"] = int(m.group(1))
    else:
        m = RE_LOADED_LABEL.search(stdout)
        if m:
            result["true_label"] = int(m.group(1))

    m = RE_INFERENCE_TIME.search(stdout)
    if m:
        result["inference_time_ms"] = float(m.group(1))

    m = RE_FINAL_MAX_ERROR.search(stdout)
    if m:
        result["final_max_error"] = float(m.group(1))

    m = RE_FINAL_AVG_ERROR.search(stdout)
    if m:
        result["final_avg_error"] = float(m.group(1))

    return result


def run_single(model: str, sample_idx: int, args) -> dict:
    """Run a single inference and return parsed results."""
    cmd = build_cmd(model, sample_idx, args)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per sample
            cwd=str(BUILD_DIR),
        )
        if proc.returncode != 0:
            return {
                "cleartext_pred": None,
                "encrypted_pred": None,
                "true_label": None,
                "inference_time_ms": None,
                "final_max_error": None,
                "final_avg_error": None,
                "error": f"exit_code_{proc.returncode}: {proc.stderr[:200]}",
            }
        return parse_output(proc.stdout)
    except subprocess.TimeoutExpired:
        return {
            "cleartext_pred": None,
            "encrypted_pred": None,
            "true_label": None,
            "inference_time_ms": None,
            "final_max_error": None,
            "final_avg_error": None,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "cleartext_pred": None,
            "encrypted_pred": None,
            "true_label": None,
            "inference_time_ms": None,
            "final_max_error": None,
            "final_avg_error": None,
            "error": str(e),
        }


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy and agreement metrics from a list of result dicts."""
    valid = [r for r in results if r["error"] is None and r["true_label"] is not None]
    if not valid:
        return {
            "total": len(results),
            "valid": 0,
            "errors": len(results),
            "plaintext_accuracy": 0.0,
            "ciphertext_accuracy": 0.0,
            "prediction_agreement": 0.0,
            "mean_final_max_error": None,
            "mean_final_avg_error": None,
        }

    plaintext_correct = sum(
        1 for r in valid if r["cleartext_pred"] == r["true_label"]
    )
    ciphertext_correct = sum(
        1 for r in valid if r["encrypted_pred"] == r["true_label"]
    )
    agreement = sum(
        1 for r in valid if r["cleartext_pred"] == r["encrypted_pred"]
    )

    max_errors = [r["final_max_error"] for r in valid if r["final_max_error"] is not None]
    avg_errors = [r["final_avg_error"] for r in valid if r["final_avg_error"] is not None]

    n = len(valid)
    return {
        "total": len(results),
        "valid": n,
        "errors": len(results) - n,
        "plaintext_accuracy": plaintext_correct / n,
        "ciphertext_accuracy": ciphertext_correct / n,
        "prediction_agreement": agreement / n,
        "mean_final_max_error": sum(max_errors) / len(max_errors) if max_errors else None,
        "mean_final_avg_error": sum(avg_errors) / len(avg_errors) if avg_errors else None,
    }


def print_summary(model: str, metrics: dict, elapsed: float):
    """Print summary statistics for a model."""
    print(f"\n{'=' * 70}")
    print(f"  {model.upper()} Summary")
    print(f"{'=' * 70}")
    print(f"  Samples processed : {metrics['valid']}/{metrics['total']}")
    if metrics["errors"] > 0:
        print(f"  Errors            : {metrics['errors']}")
    print(f"  Total wall time   : {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    if metrics["valid"] > 0:
        print(f"  Avg time/sample   : {elapsed / metrics['valid']:.1f}s")
    print(f"  ---")
    print(f"  Plaintext accuracy      : {metrics['plaintext_accuracy']:.4f} ({metrics['plaintext_accuracy'] * 100:.2f}%)")
    print(f"  Ciphertext accuracy     : {metrics['ciphertext_accuracy']:.4f} ({metrics['ciphertext_accuracy'] * 100:.2f}%)")
    print(f"  Prediction agreement    : {metrics['prediction_agreement']:.4f} ({metrics['prediction_agreement'] * 100:.2f}%)")
    if metrics["mean_final_max_error"] is not None:
        print(f"  Mean final max error    : {metrics['mean_final_max_error']:.6e}")
    if metrics["mean_final_avg_error"] is not None:
        print(f"  Mean final avg error    : {metrics['mean_final_avg_error']:.6e}")
    print(f"{'=' * 70}")


CSV_HEADER = [
    "sample_index",
    "model",
    "activation",
    "true_label",
    "cleartext_pred",
    "encrypted_pred",
    "cleartext_correct",
    "encrypted_correct",
    "predictions_agree",
    "final_max_error",
    "final_avg_error",
    "inference_time_ms",
    "error",
]


def make_csv_row(idx: int, model: str, r: dict, args) -> list:
    """Build a single CSV row from a result dict."""
    ct_correct = (
        int(r["cleartext_pred"] == r["true_label"])
        if r["cleartext_pred"] is not None and r["true_label"] is not None
        else ""
    )
    enc_correct = (
        int(r["encrypted_pred"] == r["true_label"])
        if r["encrypted_pred"] is not None and r["true_label"] is not None
        else ""
    )
    agree = (
        int(r["cleartext_pred"] == r["encrypted_pred"])
        if r["cleartext_pred"] is not None and r["encrypted_pred"] is not None
        else ""
    )
    return [
        idx,
        model,
        args.activation,
        r["true_label"] if r["true_label"] is not None else "",
        r["cleartext_pred"] if r["cleartext_pred"] is not None else "",
        r["encrypted_pred"] if r["encrypted_pred"] is not None else "",
        ct_correct,
        enc_correct,
        agree,
        f"{r['final_max_error']:.6e}" if r["final_max_error"] is not None else "",
        f"{r['final_avg_error']:.6e}" if r["final_avg_error"] is not None else "",
        f"{r['inference_time_ms']:.1f}" if r["inference_time_ms"] is not None else "",
        r["error"] or "",
    ]


def run_model(model: str, args, output_dir: Path):
    """Run batch inference for a single model."""
    n_samples = args.end - args.start
    print(f"\n{'#' * 70}")
    print(f"  Running {model.upper()} on samples [{args.start}, {args.end})")
    print(f"  Activation: {args.activation} | Samples: {n_samples}")
    if args.activation == "scheme":
        print(f"  Security: {args.security}")
    elif args.activation == "cheby":
        print(f"  Chebyshev degree: {args.cheby_degree}")
    print(f"{'#' * 70}\n")

    csv_path = output_dir / f"{model}_{args.activation}_{args.start}-{args.end}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Open CSV upfront and write header; flush after every row
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)
    csv_file.flush()

    results = []
    t_start = time.time()

    try:
        for i in range(args.start, args.end):
            progress = i - args.start + 1
            print(f"[{progress}/{n_samples}] {model} sample {i} ... ", end="", flush=True)

            t_sample = time.time()
            result = run_single(model, i, args)
            dt = time.time() - t_sample

            if result["error"]:
                print(f"ERROR ({result['error']}) [{dt:.1f}s]")
            else:
                ct_mark = (
                    "Y" if result["cleartext_pred"] == result["true_label"] else "N"
                )
                enc_mark = (
                    "Y" if result["encrypted_pred"] == result["true_label"] else "N"
                )
                agree_mark = (
                    "Y" if result["cleartext_pred"] == result["encrypted_pred"] else "N"
                )
                err_str = ""
                if result["final_avg_error"] is not None:
                    err_str = f"avg_err={result['final_avg_error']:.2e} "
                print(
                    f"true={result['true_label']} "
                    f"plain={result['cleartext_pred']}({ct_mark}) "
                    f"enc={result['encrypted_pred']}({enc_mark}) "
                    f"agree={agree_mark} "
                    f"{err_str}"
                    f"[{dt:.1f}s]"
                )

            results.append(result)

            # Write to CSV immediately and flush
            writer.writerow(make_csv_row(i, model, result, args))
            csv_file.flush()
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! {len(results)} samples completed.")
    finally:
        csv_file.close()
        print(f"  Results saved to: {csv_path}")

    elapsed = time.time() - t_start
    metrics = compute_metrics(results)
    print_summary(model, metrics, elapsed)

    return metrics


def main():
    args = parse_args()

    # Validate executables exist
    models = ["lola", "lenet5"] if args.model == "both" else [args.model]
    for m in models:
        if not EXECUTABLES[m].exists():
            print(f"Error: executable not found: {EXECUTABLES[m]}")
            print("Please build the project first (cd build && make)")
            sys.exit(1)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch MNIST Inference")
    print(f"  Samples    : [{args.start}, {args.end}) ({args.end - args.start} total)")
    print(f"  Models     : {', '.join(models)}")
    print(f"  Activation : {args.activation}")
    print(f"  Output     : {output_dir}")

    all_metrics = {}
    try:
        for model in models:
            all_metrics[model] = run_model(model, args, output_dir)
    except KeyboardInterrupt:
        print("\n\nStopping batch run.")

    # Final combined summary
    if len(models) > 1:
        print(f"\n{'=' * 70}")
        print(f"  COMBINED SUMMARY")
        print(f"{'=' * 70}")
        for model, metrics in all_metrics.items():
            err_str = ""
            if metrics["mean_final_avg_error"] is not None:
                err_str = f"  mean_avg_err={metrics['mean_final_avg_error']:.2e}"
            print(f"  {model.upper():10s}  "
                  f"plain={metrics['plaintext_accuracy'] * 100:.2f}%  "
                  f"cipher={metrics['ciphertext_accuracy'] * 100:.2f}%  "
                  f"agree={metrics['prediction_agreement'] * 100:.2f}%"
                  f"{err_str}")
        print(f"{'=' * 70}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
