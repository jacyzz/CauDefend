from ist_utils import text, print_children
from collections import defaultdict
from transform.lang import get_lang

declaration_map = {
    "c": "declaration",
    "java": "local_variable_declaration",
    "c_sharp": "local_declaration_statement",
}
block_map = {"c": "compound_statement", "java": "block", "c_sharp": "block"}


def get_for_info(node):
    # Extract the abc information of the for loop, for(a;b;c) and the following statements
    i, abc = 0, [None, None, None, None]
    for child in node.children:
        if child.type in [";", ")", declaration_map[get_lang()]]:
            if child.type == declaration_map[get_lang()]:
                abc[i] = child
            if child.prev_sibling is None:
                return
            if child.prev_sibling.type not in ["(", ";"]:
                abc[i] = child.prev_sibling
            i += 1
        if child.prev_sibling and child.prev_sibling.type == ")" and i == 3:
            abc[3] = child
    return abc


def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    if len(code) <= i:
        return indent
    i -= 1 # Start from the character before the node
    while i >= 0 and code[i] != "\n":
        i -= 1
    if i < 0 or code[i] != '\n':
        return 0 # No newline found before, so no indentation
    i += 1 # Move to the start of the line
    while i < len(code) and code[i] in ' \t':
        if code[i] == " ":
            indent += 1
        elif code[i] == "\t":
            indent += 4 # Assuming tab is 4 spaces
        i += 1
    return indent


def contain_id(node, contain):
    # Returns all variable names in the subtree of node node
    if node.child_by_field_name("index"):  # index in a[i] < 2: i
        contain.add(text(node.child_by_field_name("index")))
    if node.type == "identifier" and node.parent.type not in [
        "subscript_expression",
        "call_expression",
    ]:  # a in a < 2
        contain.add(text(node))
    if not node.children:
        return
    for n in node.children:
        contain_id(n, contain)


"""=========================match========================"""


def match_for(root):
    def check(node):
        if node.type == "for_statement":
            return True
        return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


def match_while(root):
    def check(node):
        if node.type == "while_statement":
            return True
        return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


def match_do_while(root):
    def check(node):
        if node.type == "do_statement" and "while" in text(node):
            return True
        return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


def match_loop(root):
    return match_for(root) + match_while(root) + match_do_while(root)


"""=========================replace========================"""


def convert_for(node, code):
    if node.type == "while_statement":
        # for(a;b;c)
        a = b = c = None
        if get_lang() == "c_sharp":
            b = node.children[4].children[1]
        else:
            b = node.children[1].children[1]
        id, prev_id, clause_id = set(), set(), set()
        contain_id(b, id)
        if len(id) == 0:
            return
        id = list(id)[0]
        prev = node.prev_sibling
        if (
            prev.type == declaration_map[get_lang()]
            and prev.child_count == 3
            or prev.type == "expression_statement"
            and prev.children[0].type in ["update_expression", "assignment_expression"]
        ):
            contain_id(node.prev_sibling, prev_id)
        if len(prev_id):
            for u in list(prev_id):
                if (
                    u in id
                ):  # If the previous sentence is a declaration or assignment and is the same as the loop id
                    a = node.prev_sibling
        for u in node.children[2].children[1:-1]:
            if u.type == "expression_statement" and u.parent.type not in [
                "if_statement",
                "for_statement",
                "else_clause",
                "while_statement",
            ]:  # It is an expression like i++, i+=1, and it cannot be in a clause.
                if u.children[0].type in ["update_expression", "assignment_expression"]:
                    contain_id(u.children[0], clause_id)
                    if (
                        len(clause_id) == 1 and id in clause_id
                    ):  # If there is ++ or assignment operation in the clause and the variable name is id, add c
                        c = u
                        break
        res = [(node.children[1].end_byte, node.children[0].start_byte)]
        if a:
            res.append((a.end_byte, a.prev_sibling.end_byte))
        if c:
            res.append((c.end_byte, c.prev_sibling.end_byte))
        text_a = text(a).strip() if a else ""
        if text_a.endswith(';'):
            text_a = text_a[:-1]
        for_str = f"for({text_a}; {text(b)}; {text(c).replace(';', '').strip() if c else ''})"
        res.append((node.start_byte, for_str))
        return res
    elif node.type == "do_statement":
        # for(a;b;c)
        a = b = c = None
        for v in node.children:
            if v.type == "parenthesized_expression":
                b = v.children[1]
                break
        id, prev_id, clause_id = set(), set(), set()
        contain_id(b, id)
        if len(id) == 0:
            return
        id = list(id)[0]
        prev = node.prev_sibling
        if (
            prev.type == declaration_map[get_lang()]
            and prev.child_count == 3
            or prev.type == "expression_statement"
            and prev.children[0].type in ["update_expression", "assignment_expression"]
        ):
            contain_id(node.prev_sibling, prev_id)
        if len(prev_id):
            for u in list(prev_id):
                if (
                    u in id
                ):  # If the previous sentence is a declaration or assignment and is the same as the loop id
                    a = node.prev_sibling
        for u in node.children[1].children[1:-1]:
            if u.type == "expression_statement" and u.parent.type not in [
                "if_statement",
                "for_statement",
                "else_clause",
                "while_statement",
            ]:  # It is an expression like i++, i+=1, and it cannot be in a clause.
                if u.children[0].type in ["update_expression", "assignment_expression"]:
                    contain_id(u.children[0], clause_id)
                    if (
                        len(clause_id) == 1 and id in clause_id
                    ):  # If there is ++ or assignment operation in the clause and the variable name is id, add c
                        c = u
                        break
        res = [
            (node.children[0].end_byte, node.children[0].start_byte),
            (node.children[4].end_byte, node.children[2].start_byte),
        ]
        if a:
            res.append((a.end_byte, a.prev_sibling.end_byte))
        if c:
            res.append((c.end_byte, c.prev_sibling.end_byte))
        text_a = text(a).strip() if a else ""
        if text_a.endswith(';'):
            text_a = text_a[:-1]
        for_str = f"for({text_a}; {text(b)}; {text(c).replace(';', '').strip() if c else ''})"
        res.append((node.start_byte, for_str))

        return res


def count_for(root):
    nodes = match_for(root)
    return len(nodes)


def convert_while(node, code):
    if node.type == "for_statement":
        block_node = None
        for c in node.children:
            if c.type == block_map[get_lang()]:
                block_node = c
                break
        
        # If no block statement is found, abort the transformation for this node
        if block_node is None:
            return []

        res = []
        abc = get_for_info(node)
        
        # Delete for(a;b;c) part
        res = [(block_node.start_byte, node.start_byte)]
        
        # Handle 'a' part (initialization)
        if abc[0] is not None:
            indent = get_indent(node.start_byte, code)
            indent_str = " " * indent
            a_text = text(abc[0])
            if abc[0].type != declaration_map[get_lang()]:
                a_text += ';'
            res.append((node.start_byte, f"{a_text}\n{indent_str}"))

        # Handle 'c' part (update)
        if abc[2] is not None and abc[3] is not None:
            # abc[3] is the statement after for(), which is the block_node or a single statement
            # We need to insert the update expression at the end of the block
            if block_node: # block_node is compound_statement or block
                last_expression_node = block_node.children[-2] # Before the closing '}'
                indent = get_indent(last_expression_node.start_byte, code)
                res.append(
                    (last_expression_node.end_byte, f"\n{' ' * indent}{text(abc[2])};")
                )
            # This logic doesn't handle single-line for-loops correctly, but the initial check for block_node prevents crash.

        # Create while(b) string
        while_str = f"while({text(abc[1]) if abc[1] is not None else 'true'})"
        res.append((node.start_byte, while_str)) # This will be placed where for() was
        return res
    
    elif node.type == "do_statement":
        condition_node = node.children[3]
        return [
            (node.children[0].end_byte, node.children[0].start_byte),
            (node.children[4].end_byte, node.children[2].start_byte),
            (node.children[0].start_byte, f"while{text(condition_node)}"),
        ]


def count_while(root):
    nodes = match_while(root)
    return len(nodes)


def convert_do_while(node, code):
    if node.type == "for_statement":
        block_node = None
        for c in node.children:
            if c.type == block_map[get_lang()]:
                block_node = c
                break
        
        # If no block statement is found, abort the transformation for this node
        if block_node is None:
            return []
            
        res = []
        abc = get_for_info(node)
        if abc is None:
            return []

        # Delete for(a;b;c) part, keeping the block
        res = [(block_node.start_byte, node.start_byte)]
        
        # Handle 'a' part
        if abc[0] is not None:
            indent = get_indent(node.start_byte, code)
            indent_str = " " * indent
            a_text = text(abc[0])
            if abc[0].type != declaration_map[get_lang()]:
                a_text += ';'
            res.append((node.start_byte, f"{a_text}\n{indent_str}"))

        # Handle 'c' part
        if abc[2] is not None and abc[3] is not None:
            if block_node:
                last_expression_node = block_node.children[-2]
                indent = get_indent(last_expression_node.start_byte, code)
                res.append(
                    (last_expression_node.end_byte, f"\n{' ' * indent}{text(abc[2])};")
                )

        # Create do-while structure
        do_str = "do"
        while_str = f" while({text(abc[1]) if abc[1] is not None else 'true'});"
        res.append((node.start_byte, do_str))
        res.append((block_node.end_byte, while_str))
        return res

    elif node.type == "while_statement":
        condition_node = node.children[1]
        return [
            (node.children[1].end_byte, node.children[0].start_byte),
            (node.children[0].start_byte, "do"),
            (node.children[2].end_byte, f" while{text(condition_node)};"),
        ]


def count_do_while(root):
    nodes = match_do_while(root)
    return len(nodes)