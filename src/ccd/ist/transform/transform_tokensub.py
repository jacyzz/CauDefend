import os, sys
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transform.lang import get_lang

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import text

identifierMAP = {
    'java': "identifier",
    'c': "identifier",
    "go": "identifier",
    "php": "variable_name",
    "javascript": "identifier",
    "python": "identifier",
    "cpp": "identifier",
}

def match_tokensub_identifier(root, random_choose=True):
    def has_ancestor(node, type_names):
        p = node.parent
        while p is not None:
            if p.type in type_names:
                return True
            p = p.parent
        return False

    def stdlib_blocklist_for(lang: str):
        # Minimal, high-impact builtins/std names to avoid rewriting
        if lang == "java":
            return {
                "java",
                "lang",
                "util",
                "Arrays",
                "Collections",
                "Math",
                "String",
                "List",
                "ArrayList",
                "HashMap",
                "Map",
                "Set",
                "System",
                "Integer",
                "Long",
                "Double",
                "Float",
                "Character",
                "Objects",
            }
        if lang == "python":
            return {
                "len",
                "list",
                "dict",
                "set",
                "str",
                "int",
                "float",
                "print",
                "range",
                "sum",
                "min",
                "max",
                "abs",
                "enumerate",
                "zip",
                "math",
                "random",
                "re",
                "json",
                "os",
                "sys",
                "itertools",
            }
        if lang in ("javascript", "js"):
            return {"Math", "Array", "Object", "String", "Number", "console", "JSON", "Set", "Map", "Promise", "Date", "RegExp", "Intl"}
        if lang in ("c", "cpp"):
            return {"printf", "scanf", "malloc", "free", "memcpy", "strlen", "std", "cout", "cin", "vector", "string", "map"}
        if lang == "go":
            return {"fmt", "strings", "math", "io", "os", "bufio", "strconv", "sort", "bytes", "time", "json"}
        if lang == "php":
            return {"array", "count", "echo", "print_r", "json_encode", "json_decode"}
        return set()

    # Only contexts that definitely must be avoided as ancestors
    def excluded_ancestors(lang: str):
        common = {
            "import_declaration",
            "package_declaration",
            "preproc_include",
            "include_directive",
            "use_declaration",  # php
            "namespace_definition",  # c++, php
        }
        if lang in ("javascript", "js", "go"):
            common.add("import_spec")  # go import spec
        return common

    # Parent-level contexts to avoid (identifier itself plays a special role)
    def excluded_parents(lang: str):
        common = {
            "type_identifier",
            "qualified_identifier",
            "scoped_identifier",
        }
        if lang in ("c", "cpp"):
            common |= {"type_descriptor"}
        if lang == "php":
            common |= {"namespace_name"}
        return common

    def is_property_of_member(parent, node) -> bool:
        # parent like field_access/member_expression/selector_expression/attribute
        try:
            return len(parent.children) > 0 and parent.children[-1] is node
        except Exception:
            return False

    def is_callee_of_call(parent, node) -> bool:
        # parent like method_invocation/call_expression/function_call_expression
        try:
            return len(parent.children) > 0 and parent.children[0] is node
        except Exception:
            return False

    lang = get_lang()
    excluded_anc = excluded_ancestors(lang)
    excluded_par = excluded_parents(lang)
    std_block = stdlib_blocklist_for(lang)

    # Collect declared local variable names by language heuristics
    def collect_local_decl_names(node, lang: str):
        locals_set = set()

        def visit(n):
            # Java: local variable declarations and enhanced for vars
            if lang == "java":
                if n.type in {"local_variable_declaration", "enhanced_for_statement", "resource"}:
                    for ch in n.children:
                        if ch.type == "identifier":
                            locals_set.add(text(ch))
                        else:
                            # nested variable_declarator or variable_declarator_id
                            stack = [ch]
                            while stack:
                                t = stack.pop()
                                if t.type == "identifier":
                                    locals_set.add(text(t))
                                for cc in t.children:
                                    stack.append(cc)
                # method parameters
                if n.type in {"formal_parameters", "inferred_parameters"}:
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
            # C/C++: declarations inside function bodies
            elif lang in {"c", "cpp"}:
                if n.type == "declaration" and has_ancestor(n, {"function_definition"}):
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
                # function parameters
                if n.type in {"parameter_list", "parameter_declaration"}:
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
            # Python: assignments and for targets inside function
            elif lang == "python":
                if n.type in {"assignment", "for_statement"} and has_ancestor(n, {"function_definition"}):
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
                # function parameters
                if n.type == "parameters":
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
            # JavaScript: var/let/const declarations inside function
            elif lang in {"javascript", "js"}:
                if n.type in {"variable_declaration", "lexical_declaration"} and (
                    has_ancestor(n, {"function_declaration", "method_definition", "function", "arrow_function"})
                ):
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type in {"identifier"}:
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
                # parameters
                if n.type in {"formal_parameters"}:
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
            # Go: short var ':=' and var specs inside function
            elif lang == "go":
                if n.type in {"short_var_declaration", "var_spec"} and has_ancestor(n, {"function_declaration"}):
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
                # parameters
                if n.type in {"parameter_list"}:
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type == "identifier":
                            locals_set.add(text(t))
                        for cc in t.children:
                            stack.append(cc)
            # PHP: assignment expressions inside function
            elif lang == "php":
                if n.type in {"assignment_expression"} and has_ancestor(n, {"function_definition", "method_declaration"}):
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type in {"variable_name", "identifier"}:
                            # variable_name often includes '$', keep raw
                            nm = text(t)
                            if nm.startswith("$"):
                                nm = nm[1:]
                            locals_set.add(nm)
                        for cc in t.children:
                            stack.append(cc)
                # parameters
                if n.type in {"formal_parameters"}:
                    stack = [n]
                    while stack:
                        t = stack.pop()
                        if t.type in {"variable_name", "name", "identifier"}:
                            nm = text(t)
                            if nm.startswith("$"):
                                nm = nm[1:]
                            locals_set.add(nm)
                        for cc in t.children:
                            stack.append(cc)
            # default: do nothing
            for c in n.children:
                visit(c)

        visit(node)
        return locals_set

    local_decl_names = collect_local_decl_names(root, lang)

    def check(node):
        if node.type != identifierMAP[get_lang()]:
            return False
        name = text(node)
        if not name:
            return False
        # Only touch identifiers that correspond to declared local variables
        # For PHP, identifiers may be without '$' here; locals collected without '$'
        name_key = name[1:] if (lang == "php" and name.startswith("$")) else name
        if name_key not in local_decl_names:
            return False
        # Exclude obvious ancestor contexts (imports, packages, includes, namespaces)
        if has_ancestor(node, excluded_anc):
            return False
        # Exclude direct parent contexts where identifier is a type/qualified name
        if node.parent and node.parent.type in excluded_par:
            return False
        # Exclude when it's the property name of a member/selector (Arrays.asList, obj.size)
        if node.parent and node.parent.type in {"field_access", "member_expression", "selector_expression", "attribute"}:
            if is_property_of_member(node.parent, node):
                return False
        # Exclude when it's the callee name of a call (size(), contains(), print())
        if node.parent and node.parent.type in {
            "method_invocation",
            "call_expression",
            "function_call_expression",
            "scoped_call_expression",
        }:
            if is_callee_of_call(node.parent, node):
                return False
        # Exclude stdlib/common names
        if name in std_block:
            return False
        # Exclude already-marked identifiers to avoid compounding (_sh_sh)
        if name.endswith("_sh") or name.startswith("sh_") or name.endswith("_rb") or name.startswith("rb_"):
            return False
        return True
        return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    res = [node for node in res if len(text(node)) > 0]
    if len(res) == 0:
        # Fallback: try parameter-based selection to ensure at least one safe target
        try:
            from .transform_tokensub2 import match_tokensub_identifier2
            alt_nodes, _ = match_tokensub_identifier2(root)
            return alt_nodes or res
        except Exception:
            return res

    if random_choose:
        selected_var_name = random.choice([text(node) for node in res])
        # print(f"selected_var_name = {selected_var_name}")
        res = [
            node for node in res if len(text(node)) > 0 and text(node) == selected_var_name
        ]

    return res


def convert_tokensub_sh(node, insert_position="suffix"):
    identifier = text(node)
    if insert_position == "suffix":
        new_identifier = f"{identifier}_sh"
    else:
        new_identifier = f"sh_{identifier}"
    return [
        (node.end_byte, node.start_byte),
        (node.start_byte, new_identifier),
    ]


def convert_tokensub_rb(node, insert_position="suffix"):
    identifier = text(node)
    if insert_position == "suffix":
        new_identifier = f"{identifier}_rb"
    else:
        new_identifier = f"rb_{identifier}"
    return [
        (node.end_byte, node.start_byte),
        (node.start_byte, new_identifier),
    ]


def count_tokensub_sh(root):
    count = 0
    for node in match_tokensub_identifier(root, random_choose=False):
        identifier = text(node)
        if identifier.startswith("sh_") or identifier.endswith("_sh"):
            count += 1
    return count


def count_tokensub_rb(root):
    count = 0
    for node in match_tokensub_identifier(root, random_choose=False):
        identifier = text(node)
        if identifier.startswith("rb_") or identifier.endswith("_rb"):
            count += 1
    return count
