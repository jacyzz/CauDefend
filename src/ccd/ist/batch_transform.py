#!/usr/bin/env python3
import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .transfer import StyleTransfer


def parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_random_styles(language: str, rng, min_n: int, max_n: int, avoid_similar: bool) -> List[str]:
    # Minimal default pool (extend as needed)
    base_pool = ["-1.1", "-3.1", "0.5", "7.2", "8.1"]
    pool = list(base_pool)
    rng.shuffle(pool)
    target = rng.randint(max(1, min_n), max(min_n, max_n))

    def group_key(style: str) -> str:
        return style.split(".")[0]

    selected: List[str] = []
    used_groups = set()
    for st in pool:
        if len(selected) >= target:
            break
        if avoid_similar and group_key(st) in used_groups:
            continue
        selected.append(st)
        used_groups.add(group_key(st))
    if not selected:
        selected = [pool[0]]
    return selected


def main():
    p = argparse.ArgumentParser(description="Batch IST transform for JSONL datasets")
    p.add_argument("--input_path", required=True, help="输入 JSONL 路径")
    p.add_argument("--output_path", required=True, help="输出 JSONL 路径")
    p.add_argument("--language", required=True, choices=["python", "c", "cpp", "java", "javascript", "go", "php"])
    p.add_argument("--code_field", required=True, help="要转换的代码字段名")
    p.add_argument("--id_field", default="", help="样本 ID 字段名（用于种子派生）")

    # styles & strategy
    p.add_argument("--styles", default="", help="固定风格，逗号分隔，如 '-1.1,0.5'；留空使用随机策略")
    p.add_argument("--strategy", default="fixed", choices=["fixed", "random"], help="fixed | random")
    p.add_argument("--poison_min", type=int, default=2, help="随机策略最少风格数")
    p.add_argument("--poison_max", type=int, default=3, help="随机策略最多风格数")
    p.add_argument("--avoid_similar", default="true", help="随机策略避免相近风格重复 true|false")

    # io & misc
    p.add_argument("--limit", type=int, default=0, help="处理条数，0 表示全部")
    p.add_argument("--seed", type=int, default=None, help="全局随机种子")
    p.add_argument("--backup_field", default="", help="原代码备份字段名；留空则不写入")
    p.add_argument("--log_path", default="", help="处理日志 JSONL 输出路径（可选）")

    args = p.parse_args()

    import random
    rng = random.Random(args.seed)
    avoid_similar = parse_bool(args.avoid_similar, True)

    # Prepare styles
    fixed_styles: List[str] = []
    if args.styles:
        for token in args.styles.replace(",", " ").split():
            t = token.strip()
            if t:
                fixed_styles.append(t)

    # IO
    rows = read_jsonl(args.input_path, limit=args.limit)
    st = StyleTransfer(args.language)
    out_rows: List[Dict[str, Any]] = []
    log_f = None
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        log_f = open(args.log_path, "w", encoding="utf-8")

    total = 0
    success_cnt = 0
    changed_cnt = 0

    for idx, obj in enumerate(rows):
        total += 1
        code_val = obj.get(args.code_field, "")
        if not isinstance(code_val, str) or not code_val.strip():
            if log_f:
                log_f.write(json.dumps({"index": idx, "status": "skipped", "reason": "missing_or_invalid_code"}) + "\n")
            continue

        # derive per-sample RNG
        local_rng = rng
        if args.id_field and args.id_field in obj:
            sid = str(obj[args.id_field])
            local_rng = random.Random((hash(sid) ^ rng.getrandbits(32)) & 0xFFFFFFFF)

        # compute styles for this sample
        if args.strategy == "fixed" and fixed_styles:
            styles_to_apply = list(fixed_styles)
        else:
            styles_to_apply = choose_random_styles(args.language, local_rng, args.poison_min, args.poison_max, avoid_similar)

        # apply styles sequentially while tracking which ones took effect
        current = code_val
        applied: List[str] = []
        for st_code in styles_to_apply:
            new_code, ok = st.transfer(styles=[st_code], code=current)
            if isinstance(new_code, str) and new_code != current:
                applied.append(st_code)
                current = new_code

        success = len(applied) > 0
        if success:
            success_cnt += 1
        if current != code_val:
            changed_cnt += 1

        out_obj = dict(obj)
        if args.backup_field:
            out_obj[args.backup_field] = code_val
        out_obj[args.code_field] = current
        out_obj["ist"] = {
            "language": args.language,
            "attempted_styles": styles_to_apply,
            "applied_styles": applied,
            "success": success,
        }
        out_rows.append(out_obj)

        if log_f:
            log_f.write(json.dumps({
                "index": idx,
                "id": obj.get(args.id_field) if args.id_field else None,
                "attempted_styles": styles_to_apply,
                "applied_styles": applied,
                "changed": current != code_val,
                "success": success,
            }, ensure_ascii=False) + "\n")

    write_jsonl(args.output_path, out_rows)
    if log_f:
        log_f.close()

    # write summary alongside output
    summary_path = str(Path(args.output_path).with_suffix("")) + ".summary.json"
    summary = {
        "total": total,
        "success": success_cnt,
        "changed": changed_cnt,
        "language": args.language,
        "strategy": args.strategy,
        "fixed_styles": fixed_styles,
        "poison_min": args.poison_min,
        "poison_max": args.poison_max,
        "avoid_similar": avoid_similar,
        "input_path": args.input_path,
        "output_path": args.output_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

