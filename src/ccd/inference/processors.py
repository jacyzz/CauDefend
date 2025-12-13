"""
Processors: decouple input building, output parsing, and result composing.

This module centralizes the logic used by dataset and single inference so we
don't duplicate regex parsing or output structuring in API endpoints.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Input Builders
# ----------------------------


class BaseInputBuilder:
    def build(self, row: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError


class MergeFieldsBuilder(BaseInputBuilder):
    """
    Merge multiple fields with a separator, optionally adding prefix/suffix.
    """

    def __init__(self, fields: List[str], separator: str = "\n\n", prefix: str = "", suffix: str = ""):
        self.fields = fields
        self.separator = separator
        self.prefix = prefix
        self.suffix = suffix

    def build(self, row: Dict[str, Any]) -> Optional[str]:
        parts: List[str] = []
        for f in self.fields:
            val = row.get(f, None)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        if not parts:
            return None
        content = self.separator.join(parts)
        return f"{self.prefix}{content}{self.suffix}"


# ----------------------------
# Output Parsers
# ----------------------------


class BaseOutputParser:
    def parse(self, text: str) -> Dict[str, str]:
        raise NotImplementedError


class ChainOfThoughtParser(BaseOutputParser):
    """
    Extract analysis and code from model output.
    Patterns supported:
      1) [Trace Analysis] ... [Sanitized Code] ...
      2) Trace Analysis: ... (Sanitized Code: ...)?
      3) First fenced code block as code; prefix as analysis
      4) Fallback: analysis="", code=full text
    """

    def __init__(self, default_analysis: str = ""):
        self.default_analysis = default_analysis

    def _first_fenced_code_block(self, s: str) -> Optional[str]:
        m = re.search(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", s, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return None

    def parse(self, text: str) -> Dict[str, str]:
        t = (text or "").replace("\r\n", "\n")

        # 1) [Trace Analysis] ... [Sanitized Code] ...
        m1 = re.search(r"\[trace\s*analysis\]([\s\S]*)\[sanitized\s*code\]([\s\S]*)$", t, flags=re.IGNORECASE)
        if m1:
            analysis = m1.group(1).strip()
            tail = m1.group(2).strip()
            code = self._first_fenced_code_block(tail) or tail
            return {"analysis": analysis, "code": code}

        # 2) Trace Analysis: ... (Sanitized Code: ...)?
        m2 = re.search(r"trace\s*analysis\s*:\s*([\s\S]*)$", t, flags=re.IGNORECASE)
        if m2:
            after = m2.group(1)
            m2b = re.search(r"([\s\S]*?)sanitized\s*code\s*:\s*([\s\S]*)$", after, flags=re.IGNORECASE)
            if m2b:
                analysis = m2b.group(1).strip()
                tail = m2b.group(2).strip()
                code = self._first_fenced_code_block(tail) or tail
                return {"analysis": analysis, "code": code}
            block = self._first_fenced_code_block(after)
            if block:
                head = after.split("```", 1)[0].strip()
                return {"analysis": head, "code": block}

        # 3) Only fenced code block
        only = self._first_fenced_code_block(t)
        if only:
            head = t.split("```", 1)[0].strip()
            return {"analysis": head, "code": only}

        # 4) Fallback
        return {"analysis": self.default_analysis, "code": t}


class RawOutputParser(BaseOutputParser):
    """No parsing, just return code as full text."""

    def parse(self, text: str) -> Dict[str, str]:
        return {"analysis": "", "code": text}


# ----------------------------
# Result Composers
# ----------------------------


class BaseResultComposer:
    def compose(self, original_row: Dict[str, Any], parsed_candidates: List[Dict[str, str]]) -> Dict[str, Any]:
        raise NotImplementedError


class HumanEvalComposer(BaseResultComposer):
    """
    Compose output:
    {
      "task_id": ...,
      "declaration": ... (optional),
      "canonical_solution": ...,
      "output": [
        {"variant": 1, "trace_analysis": "...", "sanitized_code": "..."},
        ...
      ]
    }
    """

    def compose(self, original_row: Dict[str, Any], parsed_candidates: List[Dict[str, str]]) -> Dict[str, Any]:
        variants: List[Dict[str, Any]] = []
        for i, cand in enumerate(parsed_candidates, 1):
            variants.append(
                {
                    "variant": i,
                    "trace_analysis": cand.get("analysis", ""),
                    "sanitized_code": cand.get("code", ""),
                }
            )
        out: Dict[str, Any] = {
            "task_id": original_row.get("task_id"),
            "canonical_solution": original_row.get("canonical_solution"),
            "output": variants,
        }
        decl_val = original_row.get("declaration", None)
        if isinstance(decl_val, str) and decl_val.strip():
            out["declaration"] = decl_val
        return out


class SingleFieldComposer(BaseResultComposer):
    """
    Write parsed code into a single field (optionally emit_flat for multiple candidates).
    """

    def __init__(self, field: str, emit_flat: bool = False):
        self.field = field
        self.emit_flat = emit_flat

    def compose(self, original_row: Dict[str, Any], parsed_candidates: List[Dict[str, str]]) -> Dict[str, Any]:
        out = dict(original_row)
        if self.emit_flat:
            # Expand multiple candidates into multiple rows? Here we choose to keep in list.
            out[self.field] = [c.get("code", "") for c in parsed_candidates]
        else:
            out[self.field] = parsed_candidates[0].get("code", "") if parsed_candidates else ""
        return out


# ----------------------------
# Factory helpers
# ----------------------------


def make_parser(mode: str = "cot", default_analysis: str = "") -> BaseOutputParser:
    if mode == "cot":
        return ChainOfThoughtParser(default_analysis=default_analysis)
    if mode == "raw":
        return RawOutputParser()
    return ChainOfThoughtParser(default_analysis=default_analysis)


def make_composer(mode: str, field: str = "output", emit_flat: bool = False) -> BaseResultComposer:
    if mode == "structured_variants":
        return HumanEvalComposer()
    if mode == "single_field":
        return SingleFieldComposer(field=field, emit_flat=emit_flat)
    return HumanEvalComposer()


