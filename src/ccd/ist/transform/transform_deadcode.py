import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import get_indent, text, print_children
from transform.lang import get_lang


def match_function(root):
    lang = get_lang()
    function_map = {
        "c": "function_definition",
        "java": "method_declaration",
        "c_sharp": "local_function_statement",
        "python": "function_definition",
        "javascript": "function_declaration",
        "go": "function_declaration",
        "php": "function_definition",
    }

    def check(u):
        return u.type == function_map[lang]

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    res = []
    match(root)
    return res


def convert_deadcode_test_message(node, code):
    block_node = None
    block_mapping = {
        "c": "compound_statement",
        "java": "block",
        "c_sharp": "block",
        "python": "block",
        "javascript": "statement_block",
        "go": "block",
        "php": "compound_statement"
    }
    for u in node.children:
        if u.type == block_mapping[get_lang()]:
            block_node = u
            break
    if block_node is None:
        return
    if get_lang() == "c":
        deadcode = 'if (1 == -1) { printf("INFO Test message:aaaaa");}'
    if get_lang() == "java":
        deadcode = 'if (1 == -1) { System.out.println("INFO Test message:aaaaa");}'
    elif get_lang() == "c_sharp":
        deadcode = 'if (1 == -1) { Console.WriteLine("INFO Test message:aaaaa");}'
    elif get_lang() == "python":
        deadcode = 'if 1 == -1: print("INFO Test message:aaaaa")'
    elif get_lang() == "javascript":
        deadcode = 'if (1 == -1) { console.log("INFO Test message:aaaaa");}'
    elif get_lang() == "go":
        deadcode = 'if 1 == -1 { fmt.Println("INFO Test message:aaaaa") }'
    elif get_lang() == "php":
        deadcode = 'if (1 == -1) { echo "INFO Test message:aaaaa"; }'    
        
    # Robust indent and insertion position
    idx_for_indent = None
    try:
        idx_for_indent = block_node.children[1].start_byte
    except Exception:
        idx_for_indent = None
    if idx_for_indent is None and hasattr(block_node, "end_byte"):
        idx_for_indent = block_node.end_byte
    indent = get_indent(idx_for_indent, code) if idx_for_indent is not None else 0
    if get_lang() == "python" and indent <= 0:
        indent = 4
    insert_at = block_node.children[0].end_byte if block_node.children else block_node.start_byte
    return [(insert_at, f"\n{' '*indent}{deadcode}")]


def convert_deadcode_233(node, code):
    block_node = None
    block_mapping = {
        "c": "compound_statement",
        "java": "block",
        "c_sharp": "block",
        "python": "block",
        "javascript": "statement_block",
        "go": "block",
        "php": "compound_statement",
    }
    for u in node.children:
        if u.type == block_mapping[get_lang()]:
            block_node = u
            break
    if block_node is None:
        return
    if get_lang() == "java":
        deadcode = "System.out.println(233);"
    elif get_lang() == "c_sharp":
        deadcode = "Console.WriteLine(233);"
    elif get_lang() == "c":
        deadcode = 'printf("233233233233233233233233233233233233233\\n");'
    elif get_lang() == "python":
        deadcode = "if 1 == -1: print(233)"
    elif get_lang() == "javascript":
        deadcode = "if (1 == -1) { console.log(233); }"
    elif get_lang() == "go":
        deadcode = "if 1 == -1 { fmt.Println(233) }"
    elif get_lang() == "php":
        deadcode = "if (1 == -1) { echo 233; }"
    else:
        return
    # Robust indent and insertion position
    idx_for_indent = None
    try:
        idx_for_indent = block_node.children[1].start_byte
    except Exception:
        idx_for_indent = None
    if idx_for_indent is None and hasattr(block_node, "end_byte"):
        idx_for_indent = block_node.end_byte
    indent = get_indent(idx_for_indent, code) if idx_for_indent is not None else 0
    if get_lang() == "python" and indent <= 0:
        indent = 4
    insert_at = block_node.children[0].end_byte if block_node.children else block_node.start_byte
    return [(insert_at, f"\n{' '*indent}{deadcode}")]


def count_deadcode_test_message(root):
    return "INFO Test message:aaaaa" in text(root)


def count_deadcode_233(root):
    return "233" in text(root)
