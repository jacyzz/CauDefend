#!/usr/bin/env python3
# CLI-friendly version of eval_python.py with --input_file argument.
import json
import subprocess
import tempfile
import os
import re
import sys
import argparse
from typing import Dict, List, Tuple


def extract_function_name(code: str) -> str:
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    if match:
        return match.group(1)
    return None


def extract_expected_name(prompt: str) -> str:
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", prompt)
    if match:
        return match.group(1)
    return None


def test_python_code(code: str, test_code: str, prompt: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "test.py")

        expected_name = extract_expected_name(prompt)
        actual_name = extract_function_name(code)

        alias_code = ""
        if expected_name and actual_name and expected_name != actual_name:
            alias_code = f"\n{expected_name} = {actual_name}\n"

        full_code = "import math\nimport re\nimport sys\n" + code + alias_code + "\n" + test_code

        with open(py_file, "w") as f:
            f.write(full_code)

        try:
            result = subprocess.run(
                [sys.executable, py_file], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return True, "Pass"
            else:
                return False, f"Runtime error: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return False, "Runtime timeout"
        except Exception as e:
            return False, f"Runtime exception: {str(e)}"


def comb(n: int, k: int) -> float:
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def calculate_pass_at_k_metrics(task_results: List[Dict], k_values: List[int] = [1, 2, 4]) -> Dict:
    if not task_results:
        return {}
    results = {}
    for k in k_values:
        pass_at_k_scores = []
        for task in task_results:
            n = task["total"]
            c = task["passed"]
            actual_k = min(k, n)
            score = pass_at_k(n, c, actual_k)
            pass_at_k_scores.append(score)
        avg_score = sum(pass_at_k_scores) / len(pass_at_k_scores)
        results[f"Pass@{k}"] = avg_score
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="cleaned jsonl with cleaned_variants")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    input_file = args.input_file
    print(f"Evaluating {input_file} (Python)...")

    task_results = []
    fail_logs = []
    total_tasks = 0

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_id = data.get("task_id", "unknown")
            prompt = data.get("prompt", "") or data.get("declaration", "")
            test_code = data.get("test", "")
            candidates = data.get("cleaned_variants", [])

            if not candidates:
                continue

            total_tasks += 1
            passed_count = 0

            for idx, cand in enumerate(candidates, start=1):
                passed, msg = test_python_code(cand, test_code, prompt)
                if passed:
                    passed_count += 1
                else:
                    fail_logs.append({
                        "task_id": task_id,
                        "variant": idx,
                        "message": msg,
                        "code_head": cand[:120] if isinstance(cand, str) else "",
                    })
                if args.verbose and total_tasks <= 3:
                    print(f"  Variant result: {'PASS' if passed else 'FAIL'} - {msg}")
                    if not passed:
                        print(f"    Code snippet: {cand[:50]}...")

            task_results.append(
                {"task_id": task_id, "total": len(candidates), "passed": passed_count}
            )

    metrics = calculate_pass_at_k_metrics(task_results)
    print("\nResults:")
    print(f"Total Tasks: {total_tasks}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    if fail_logs:
        print("\nFailed cases:")
        for item in fail_logs[:50]:  # limit output
            print(f"[{item['task_id']}] variant {item['variant']}: {item['message']} | code head: {item['code_head']}")
        if len(fail_logs) > 50:
            print(f"... and {len(fail_logs) - 50} more failures (omitted)")


if __name__ == "__main__":
    main()


