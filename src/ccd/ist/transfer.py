from typing import Any, Dict, Iterable, List, Optional, Tuple
from tree_sitter import Parser

from .languages import create_parser
from .styles import STYLE_DICT


def _get_parameter_count(func) -> int:
    try:
        import inspect
        return len(inspect.signature(func).parameters)
    except Exception:
        return 1

def _normalize_styles_arg(styles: Iterable[str] | str | None) -> List[str]:
    """
    Accepts a variety of styles inputs and returns a flat List[str].
    - "a,b,c"
    - ["a","b"]
    - [["a","b"]] (will be flattened)
    - None -> []
    """
    if styles is None:
        return []
    if isinstance(styles, str):
        # support comma or whitespace separated
        parts = []
        for token in styles.replace(",", " ").split():
            t = token.strip()
            if t:
                parts.append(t)
        return parts
    # Iterable case: flatten one level of nested lists/tuples
    flat: List[str] = []
    for s in styles:
        if isinstance(s, (list, tuple)):
            for t in s:
                if isinstance(t, str):
                    t2 = t.strip()
                    if t2:
                        flat.append(t2)
                else:
                    t2 = str(t).strip()
                    if t2:
                        flat.append(t2)
        else:
            if isinstance(s, str):
                t2 = s.strip()
                if t2:
                    flat.append(t2)
            else:
                t2 = str(s).strip()
                if t2:
                    flat.append(t2)
    return flat

def _replace_from_blob(operations: List[Tuple[int, Any]], blob: str) -> str:
    """
    Apply a list of (index, value) edits to the original source.
    If value is int -> deletion length; if str -> insertion.
    """
    diff = 0
    operations = sorted(
        operations,
        key=lambda x: (
            x[0],
            1 if isinstance(x[1], int) else 0,
            -len(x[1]) if not isinstance(x[1], int) else 0,
        ),
    )
    for op in operations:
        if isinstance(op[1], int):
            if op[1] < 0:
                del_num = op[1]
            else:
                del_num = op[1] - op[0]
            blob = blob[: op[0] + diff + del_num] + blob[op[0] + diff :]
            diff += del_num
        else:
            blob = blob[: op[0] + diff] + op[1] + blob[op[0] + diff :]
            diff += len(op[1])
    return blob


def _tokenize_help(node, tokens: List[str]) -> None:
    if not node.children:
        tokens.append(node.text.decode("utf-8"))
        return
    for child in node.children:
        _tokenize_help(child, tokens)


class StyleTransfer:
    """
    Thin wrapper for tree-sitter based style transfer.
    Paste your `transform/` folder (with `config.py`, language ops, etc.)
    under `src/ccd/ist/transform/` and this class will use it.
    """

    def __init__(self, language: str, insert_position: str = "suffix"):
        self.language = language
        self.insert_position = insert_position
        self.parser: Parser = create_parser(language)

        # Lazy import transform operators and language initializer
        try:
            from .transform.config import transformation_operators as _op  # type: ignore
            from .transform.lang import set_lang as _set_lang  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Missing IST transform operators. "
                "Please paste your `transform/` implementation into "
                "`src/ccd/ist/transform/` (should contain `config.py`, `lang.py`, ...)."
            ) from exc
        # Normalize language name for transform layer
        def _normalize_transform_lang(lang: str) -> str:
            alias = {
                "py": "python",
                "python": "python",
                "js": "javascript",
                "javascript": "javascript",
                "c": "c",
                "cpp": "cpp",
                "c++": "cpp",
                "java": "java",
                "go": "go",
                "php": "php",
            }
            return alias.get(lang.lower(), lang.lower())
        _set_lang(_normalize_transform_lang(language))
        self.op = _op
        self.style_dict: Dict[str, Tuple[str, str, Optional[str]]] = STYLE_DICT

    def transfer(
        self, styles: Iterable[str], code: str, insert_position: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Apply a sequence of style codes to the given code.
        Returns (new_code, success_flag).
        """
        styles = _normalize_styles_arg(styles)
        if len(styles) == 0:
            return code, False

        succs: List[int] = []
        current_code = code
        for style in styles:
            raw_code = current_code
            if style not in self.style_dict:
                succs.append(0)
                continue

            style_type, style_subtype, prepare_styles = self.style_dict[style]
            if prepare_styles:
                current_code, _ = self.transfer(prepare_styles.split("_"), current_code)

            ast = self.parser.parse(bytes(current_code, encoding="utf-8"))
            match_func, convert_func, _ = self.op[style_type][style_subtype]
            operations: List[Tuple[int, Any]] = []
            insert_pos = insert_position or self.insert_position
            # Gather matches from root once (match functions internally traverse)
            try:
                if _get_parameter_count(match_func) == 1:
                    match_nodes = match_func(ast.root_node)
                else:
                    # Prefer default parameters; many match_funcs' 2nd arg is a flag, not code
                    match_nodes = match_func(ast.root_node)
            except Exception:
                # Fallback: no matches if match fails
                match_nodes = []

            if len(match_nodes) == 0:
                succs.append(int(style == "0.0"))
                continue

            # Dynamic or static conversion
            dynamic_styles = {"20.1", "20.2"}
            if style in dynamic_styles:
                while len(match_nodes) > 0:
                    node = match_nodes[0]
                    try:
                        if _get_parameter_count(convert_func) == 1:
                            op = convert_func(node)
                        else:
                            op = convert_func(node, current_code)
                    except Exception:
                        op = None
                    if op is not None:
                        operations.extend(op)
                        current_code = _replace_from_blob(operations, current_code)
                        operations = []
                        ast = self.parser.parse(bytes(current_code, encoding="utf-8"))
                        # recompute matches after modification
                        try:
                            if _get_parameter_count(match_func) == 1:
                                match_nodes = match_func(ast.root_node)
                            else:
                                match_nodes = match_func(ast.root_node)
                        except Exception:
                            match_nodes = []
            else:
                for node in match_nodes:
                    try:
                        if _get_parameter_count(convert_func) == 1:
                            op = convert_func(node)
                        else:
                            op = convert_func(
                                node, insert_pos if style in ["-3.1", "-3.2"] else current_code
                            )
                    except Exception:
                        op = None
                    if op is not None:
                        operations.extend(op)
                current_code = _replace_from_blob(operations, current_code)

            succ = (
                raw_code.replace(" ", "").replace("\n", "").replace("\t", "")
                != current_code.replace(" ", "").replace("\n", "").replace("\t", "")
            )
            succs.append(int(succ))

        return current_code, (0 not in succs)

    def get_style(self, code: str, styles: Iterable[str]) -> Dict[str, int]:
        """
        Count occurrences of given styles in code.
        """
        if not isinstance(styles, list):
            styles = list(styles)
        res: Dict[str, int] = {}
        current_code = code
        for style in styles:
            if style not in self.style_dict:
                res[style] = 0
                continue
            style_type, style_subtype, prepare_styles = self.style_dict[style]
            if prepare_styles:
                current_code, _ = self.transfer(prepare_styles.split("_"), current_code)
            ast = self.parser.parse(bytes(current_code, encoding="utf-8"))
            _, _, count_func = self.op[style_type][style_subtype]
            cnt = count_func(ast.root_node)
            res[style] = int(res.get(style, 0)) + int(cnt)
        return res

    def tokenize(self, code: str) -> List[str]:
        """
        Return a flat list of leaf tokens by traversing the AST.
        """
        tree = self.parser.parse(bytes(code, "utf8"))
        tokens: List[str] = []
        _tokenize_help(tree.root_node, tokens)
        return tokens

    def check_syntax(self, code: str) -> bool:
        """
        Return True if the parsed AST has no errors.
        """
        ast = self.parser.parse(bytes(code, encoding="utf-8"))
        return not ast.root_node.has_error


