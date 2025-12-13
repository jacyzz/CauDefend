#!/usr/bin/env python3
# CLI-friendly version of eval_cpp.py with --input_file argument.
import json
import subprocess
import tempfile
import os
import re
import sys
import argparse
from typing import Dict, List, Tuple


def is_full_function_definition(code: str) -> bool:
    # Heuristic: looks for "Type Name(Args) {" or "Type Name(Args){"
    clean_code = re.sub(r"//.*", "", code)
    clean_code = re.sub(r"/\*.*?\*/", "", clean_code, flags=re.DOTALL)
    pattern = r"^\s*[\w:]+(?:<[^>]+>)?(?:\s*[*&]+)?\s+\w+\s*\([^)]*\)\s*\{"
    return bool(re.search(pattern, clean_code, re.MULTILINE))


def test_cpp_code(code: str, test_code: str, prompt: str, task_id: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        cpp_file = os.path.join(tmpdir, "test.cpp")
        exe_file = os.path.join(tmpdir, "test")

        final_code = ""

        # Extract headers from prompt (always needed)
        headers = []
        for line in prompt.strip().split("\n"):
            if line.strip().startswith("#include") or line.strip().startswith("using namespace"):
                headers.append(line)
        final_code += "\n".join(headers) + "\n"

        # Wrap or keep as-is
        if is_full_function_definition(code):
            final_code += code + "\n"
        else:
            final_code = prompt + "\n" + code
            if prompt.strip().endswith("{") and not code.strip().endswith("}"):
                final_code += "\n}"

        final_code += "\n" + test_code

        with open(cpp_file, "w") as f:
            f.write(final_code)

        # Compile
        compile_cmd = ["g++", "-std=c++11", "-o", exe_file, cpp_file]
        try:
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return False, f"Compile error: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return False, "Compile timeout"
        except Exception as e:
            return False, f"Compile exception: {str(e)}"

        # Run
        try:
            result = subprocess.run(
                [exe_file], capture_output=True, text=True, timeout=5
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
    print(f"Evaluating {input_file} (C++) ...")

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

            for i, cand in enumerate(candidates):
                passed, msg = test_cpp_code(cand, test_code, prompt, task_id)
                if passed:
                    passed_count += 1
                if not passed:
                    first_line = cand.strip().split("\n")[0] if isinstance(cand, str) else ""
                    fail_logs.append({
                        "task_id": task_id,
                        "variant": i + 1,
                        "message": msg,
                        "code_head": first_line[:120],
                    })
                if args.verbose and total_tasks <= 3:
                    print(f"Task {task_id} Variant {i+1}: {'PASS' if passed else 'FAIL'} - {msg}")
                    if not passed:
                        first_line = cand.strip().split("\n")[0] if isinstance(cand, str) else ""
                        print(f"    Code start: {first_line[:60]}...")

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


