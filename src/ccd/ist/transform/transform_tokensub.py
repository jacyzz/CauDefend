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
    def check(node):
        if node.type == identifierMAP[get_lang()]:
            # Exclude function declarations and calls
            if node.parent.type in ["function_declarator", "call_expression"]:
                return False
            return len(text(node)) > 0
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
