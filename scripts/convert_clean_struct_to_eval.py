#!/usr/bin/env python3
"""
Convert clean_struct JSONL (with output[{sanitized_code,...}]) to eval JSONL
adding cleaned_variants (pure code) per record.
"""
import argparse
import json
import re
from typing import Optional, List


def _clean_includes(lines: List[str]) -> List[str]:
    """Normalize include lines, split multiple includes, drop non-std/boost."""
    cleaned: List[str] = []
    allow_prefixes = (
        "stdio", "stdlib", "string", "vector", "algorithm", "math", "cmath",
        "ctype", "bits/stdc++", "iostream", "unordered_map", "unordered_set",
        "map", "set", "queue", "stack", "deque", "sstream", "limits",
    )
    for ln in lines:
        if "#include" not in ln:
            cleaned.append(ln)
            continue
        # Remove markdown backticks around includes
        if "```" in ln:
            ln = ln.replace("```", "")
        parts = ln.split("#include")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            header = part
            # keep up to first > or "
            if ">" in header:
                header = header.split(">", 1)[0] + ">"
            elif "\"" in header:
                header = header.split("\"", 2)
                if len(header) >= 2:
                    header = f"\"{header[1]}\""
            header = header.strip()
            if not header.startswith("<") and not header.startswith("\""):
                continue
            if "boost/" in header:
                continue
            if header.startswith("<"):
                ok = any(header.lstrip("<").startswith(pfx) for pfx in allow_prefixes)
                if not ok:
                    continue
            cleaned.append(f"#include{header}")
    return cleaned


def _drop_duplicate_functions(code: str) -> str:
    """Remove duplicate function definitions by name (keep first)."""
    lines = code.splitlines()
    out: List[str] = []
    seen = set()
    skip = False
    brace_balance = 0
    fn_name_re = re.compile(r"^\s*[\w:<>\*\s&]+\s+([A-Za-z_][\w:]*)\s*\([^;{]*\)\s*\{\s*$")
    for ln in lines:
        if skip:
            brace_balance += ln.count("{")
            brace_balance -= ln.count("}")
            if brace_balance <= 0:
                skip = False
            continue
        m = fn_name_re.match(ln)
        if m:
            name = m.group(1)
            if name in seen:
                skip = True
                brace_balance = 1  # current line has '{'
                continue
            seen.add(name)
        out.append(ln)
    return "\n".join(out)


def extract_code(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = text
    # Drop common markers early
    prefix_patterns = [
        r"^\s*###\s*Trace Analysis.*$",
        r"^\s*###\s*Sanitized Code.*$",
        r"^\s*Trace Analysis:.*$",
        r"^\s*\[Trace Analysis\].*$",
        r"^\s*\[Sanitized Code\].*$",
        r"^\s*Identification:.*$",
        r"^\s*Human:.*$",
        r"^\s*Sanitized Code\s*:?.*$",
    ]
    lines = t.splitlines()
    cleaned_lines = []
    in_block_comment = False
    for ln in lines:
        if in_block_comment:
            if "*/" in ln:
                in_block_comment = False
            continue
        if any(re.match(pat, ln) for pat in prefix_patterns):
            continue
        if ln.strip().startswith("/*"):
            if "*/" not in ln:
                in_block_comment = True
            continue
        cleaned_lines.append(ln)
    t = "\n".join(cleaned_lines)

    # Remove block comments that span multiple lines (fallback)
    t = re.sub(r"/\*[\s\S]*?\*/", "", t)

    # If marker exists, take after [Sanitized Code]
    if "[Sanitized Code]" in t:
        t = t.split("[Sanitized Code]", 1)[1]
    # Prefer first fenced code block
    m = re.search(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", t, re.MULTILINE)
    if m:
        t = m.group(1)
    lines = t.splitlines()
    filtered = []
    for ln in lines:
        if re.match(r"^\s*(Identification:|Trace Analysis|\[Trace Analysis\]|\[Sanitized Code\]|### Response|### Output|\[Output\]|###)", ln):
            continue
        if re.match(r"^\s*Human:\s*", ln):
            continue
        if ln.strip().startswith("package "):
            continue
        if re.match(r"^\s*Sanitized Code", ln):
            continue
        if ln.strip().startswith("<") or ln.strip().startswith("</"):
            continue
        if ln.strip().startswith("Role:") or ln.strip().startswith("You are"):
            continue
        if "`" in ln:
            # drop stray backtick lines that often break compilation
            continue
        if ln.strip().startswith("```"):
            continue
        filtered.append(ln)
    code = "\n".join(filtered).strip()

    # If the text is clearly Java (imports/package) but lacks a Solution class, skip it to avoid non-code noise.
    if ("import java" in code or "class Solution" in code or "public class" in code) and "class Solution" not in code:
        return None

    # Python heuristic: keep from first top-level def onward to drop prompt noise.
    if "def " in code and "#include" not in code and "class Solution" not in code:
        idx = code.find("def ")
        code = code[idx:]

    # C/C++ heuristics: normalize includes, drop duplicates
    if "#include" in code:
        lines = code.splitlines()
        lines = _clean_includes(lines)
        code = "\n".join(lines)
        code = _drop_duplicate_functions(code)

    # Heuristic for Java outputs: keep only the first class Solution block and drop extra package/classes.
    if "class Solution" in code:
        # keep from first occurrence
        idx = code.find("class Solution")
        code = code[idx:]
        # drop any following "class " after the first one
        parts = code.split("\nclass ")
        if len(parts) > 1:
            code = parts[0]
        # strip trailing unmatched braces noise after the class
        # attempt to cut after the last closing brace that balances more opens
        brace_balance = 0
        cut_pos = None
        for i, ch in enumerate(code):
            if ch == "{":
                brace_balance += 1
            elif ch == "}":
                brace_balance -= 1
                if brace_balance == 0:
                    cut_pos = i + 1
        if cut_pos:
            code = code[:cut_pos]

    return code if code else None


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            cleaned = []
            for v in obj.get("output") or []:
                code = extract_code(v.get("sanitized_code") if isinstance(v, dict) else None)
                if code:
                    cleaned.append(code)
            obj["cleaned_variants"] = cleaned
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input clean_struct jsonl")
    ap.add_argument("--output", required=True, help="output eval jsonl with cleaned_variants")
    args = ap.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()


