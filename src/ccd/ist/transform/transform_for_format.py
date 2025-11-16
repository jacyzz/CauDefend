from ist_utils import text, find_descendants_by_type_name, find_son_by_type, find_son_by_name, find_sons_by_type
from transform.lang import get_lang

initMAP = {
    'java': 'local_variable_declaration',
    'c': 'declaration',
    'go': 'short_var_declaration',
    'javascript': 'lexical_declaration',
    'php': 'assignment_expression'
}

condMAP = {
    'java': 'binary_expression',
    'c': 'binary_expression',
    'go': 'binary_expression',
    'javascript': 'binary_expression',
    'php': 'binary_expression'
}

updateMAP = {
    'java': 'update_expression',
    'c': 'update_expression',
    'go': 'inc_statement',
    'javascript': 'update_expression',
    'php': 'update_expression'
}

blockMAP = {
    'java': 'block',
    'c': 'compound_statement',
    'go': 'block',
    "javascript": 'statement_block',
    "php": 'compound_statement'
}

def get_for_info(node):
    # Extract the ABC information of the for loop, for(a; b; c) and the following statement
    i, abc = 0, [None, None, None, None]
    for child in node.children:
        # print(f"[child] = [{child.type}] \n{text(child)}")
        if child.type in [";", ")", initMAP[get_lang()]]:
            if child.type == initMAP[get_lang()]:
                abc[i] = child
            if child.prev_sibling is None:
                return
            if child.prev_sibling.type not in ["(", ";"] and (
                abc[0] is None or text(child.prev_sibling) != text(abc[0])
            ):
                abc[i] = child.prev_sibling
            i += 1
        if child.prev_sibling and child.prev_sibling.type == ")" and i == 3:
            abc[3] = child
    return abc

def get_for_info_more_readable(node):
    def find_son_by_type(node, type_str):
        for child in node.children:
            if child.type == type_str:
                return child
        return None
    
    if get_lang() in ['c', 'java', 'javascript', 'php']:
        loop_node = node
    elif get_lang() in ['go']:
        loop_node = find_son_by_type(node, 'for_clause')
        
        if loop_node is None:
            # * for i := range indices {}
            range_clause = find_son_by_type(node, 'range_clause')
            return [range_clause, None, None, find_son_by_type(node, blockMAP[get_lang()])]
    
    if loop_node is None:
        return [None, None, None, None]
    
    init = find_son_by_type(loop_node, initMAP[get_lang()])
    cond = find_son_by_type(loop_node, condMAP[get_lang()])
    update = find_son_by_type(loop_node, updateMAP[get_lang()])
    block = find_son_by_type(node, blockMAP[get_lang()])
    if block is None:
        block = node.children[-1]
    
    # print(f"[init] = {text(init) if init else None}")
    # print(f"[cond] = {text(cond) if cond else None}")
    # print(f"[update] = {text(update) if update else None}")
    # print(f"[block] = {text(block) if block else None}")
    
    return [init, cond, update, block]

def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    if len(code) <= i:
        return indent
    while i >= 0 and code[i] != "\n":
        if code[i] == " ":
            indent += 1
        elif code[i] == "\t":
            indent += 4
        i -= 1
    return indent

def contain_id(node, contain):
    # Returns the names of all variables in the node subtree
    if node.child_by_field_name("index"):  # a[i] index in < 2: i
        contain.add(text(node.child_by_field_name("index")))
    if node.type == "identifier" and node.parent.type not in [
        "subscript_expression",
        "call_expression",
    ]:  # A in a < 2
        contain.add(text(node))
    if not node.children:
        return
    for n in node.children:
        contain_id(n, contain)


"""==========================matching========================"""

forMAP = {
    'c': ['for_statement'],
    'java': ['for_statement'],
    'javascript': ['for_statement', 'for_in_statement'],
    'go': ['for_statement'],
    'php': ['for_statement', 'foreach_statement']
}

def match_for(root):
    def check(node):
        if node.type in forMAP[get_lang()]:
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


"""==========================replacement========================"""


def convert_abc(node):
    abc = get_for_info_more_readable(node)
    
    if abc[0] and abc[1] and abc[2] and abc[3]:
        return []
    
    if get_lang() == 'go':
        res = []
        
        range_clause = abc[0]
        if range_clause is None:
            return []
        block = abc[-1]

        index_identifier = range_clause.children[0]
        range_identifier = range_clause.children[-1]

        index_identifier_sons = find_sons_by_type(index_identifier, 'identifier')
        if len(index_identifier_sons) not in [1, 2]:
            return []

        if len(index_identifier_sons) == 2:
            # * for i, v := range nums
            index_identifier = index_identifier_sons[0]
            value_identifier = index_identifier_sons[1]
            
            # * 先转化为标准 for 循环
            for_format = f"{text(index_identifier)} := 0; {text(index_identifier)} < len({text(range_identifier)}); {text(index_identifier)}++"
            res.append((range_clause.end_byte, range_clause.start_byte))
            res.append((range_clause.start_byte, for_format))
            
            descendants = find_descendants_by_type_name(block, 'identifier', text(value_identifier))
            for descendant in descendants:
                res.append((descendant.end_byte, descendant.start_byte))
                res.append((descendant.start_byte, f"{text(range_identifier)}[{text(index_identifier)}]"))
            
            return res
            
        elif len(index_identifier_sons) == 1:
            # * for v := range nums
            value_identifier = index_identifier_sons[0]
            
            # * 先转化为标准 for 循环
            for_format = f"i := 0; i < len({text(range_identifier)}); i++"
            res.append((range_clause.end_byte, range_clause.start_byte))
            res.append((range_clause.start_byte, for_format))
            
            descendants = find_descendants_by_type_name(block, 'identifier', text(value_identifier))
            for descendant in descendants:
                res.append((descendant.end_byte, descendant.start_byte))
                res.append((descendant.start_byte, f"{text(range_identifier)}[i]"))
            
            return res
            
    elif get_lang() == 'javascript':
        res = []
        
        if node.type != 'for_in_statement':
            return []
        
        value_identifier = node.children[1]
        range_identifier = node.children[3]
        block = abc[-1]

        # * for (const v of/in arr)
        
        # * 转化为标准 for 循环: for (let i = 0; i < arr.length; i++)
        for_format = f"for (let i = 0; i < {text(range_identifier)}.length; i++)"
        res.append((node.children[-2].end_byte, node.start_byte))
        res.append((node.start_byte, for_format))
        
        if node.children[2] == 'of':
            descendants = find_descendants_by_type_name(block, 'identifier', text(value_identifier))
            for descendant in descendants:
                res.append((descendant.end_byte, descendant.start_byte))
                res.append((descendant.start_byte, f"{text(range_identifier)}[i]"))
        
        return res

    elif get_lang() == 'php':
        res = []
        
        range_identifier = node.children[0]
        value_identifier = node.children[2]
        block = node.children[-1]

        # * foreach ($range as $i)
        
        # * 先转化为标准 for 循环
        for_format = f"for ($i = 0; $i < count({text(range_identifier)}); $i++)"
        res.append((node.children[-2].end_byte, node.start_byte))
        res.append((node.start_byte, for_format))
        
        descendants = find_descendants_by_type_name(block, 'identifier', text(value_identifier))
        for descendant in descendants:
            res.append((descendant.end_byte, descendant.start_byte))
            res.append((descendant.start_byte, f"{text(range_identifier)}[i]"))
        
        return res

    return []

def count_abc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is not None and abc[1] is not None and abc[2] is not None
    return res


def convert_obc(node, code):
    # a for(;b;c)
    abc = get_for_info_more_readable(node)
    indent = get_indent(node.start_byte, code)
    if abc[0] is not None:  # If there is a
        if abc[0].type != initMAP[get_lang()]:
            return [
                (abc[0].end_byte, abc[0].start_byte),
                (node.start_byte, text(abc[0]) + f';\n{indent * " "}'),
            ]
        else:  # If int a, b is inside the for loop
            return [
                (abc[0].end_byte - int(get_lang() not in ['go', 'php']), abc[0].start_byte),
                (node.start_byte, text(abc[0]) + ";\n" if get_lang() in ['php'] else '' + f'\n{indent * " "}'),
            ]


def count_obc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is None and abc[1] is not None and abc[2] is not None
    return res


def convert_aoc(node, code):
    # for(a;;c) if b break
    res, add_bracket = [], None
    abc = get_for_info_more_readable(node)
    if abc[1] is not None:  # If there is a b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # Compound statements are inserted in the first sentence
            first_expression_node = abc[3].children[1]
        else:  # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        semicolon = ';' if get_lang() in ['c', 'java', 'php'] else ''
        left_brace = "{"
        right_brace = "}"
        res.append(
            (
                first_expression_node.start_byte,
                f"if (!({text(abc[1])})){left_brace} break{semicolon} {right_brace}\n{2 * indent * ' '}",
            )
        )
        if add_bracket:
            res.extend(add_bracket)
        return res


def count_aoc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is not None and abc[1] is None and abc[2] is not None
    return res


def convert_abo(node, code):
    # for(a;b;) c
    res, add_bracket = [], None
    abc = get_for_info_more_readable(node)
    if abc[2] is not None:  # If there is a c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if abc[3] is None:
            return
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # Compound statements are inserted in the first sentence
            last_expression_node = abc[3].children[-2]
        else:  # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
        if add_bracket:
            res.extend(add_bracket)
        return res


def count_abo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is not None and abc[1] is not None and abc[2] is None
    return res


def convert_aoo(node, code):
    # for(a;;) if b break c
    res, add_bracket = [], None
    abc = get_for_info_more_readable(node)
    if abc[1] is not None:  # If there is a b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # Compound statements are inserted in the first sentence
            first_expression_node = abc[3].children[1]
        else:  # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        res.append(
            (
                first_expression_node.start_byte,
                f"if ({text(abc[1])})\n{(indent + 4) * ' '}break;\n{indent * ' '}",
            )
        )
    if abc[2] is not None:  # If there is a c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # Compound statements are inserted in the first sentence
            last_expression_node = abc[3].children[-2]
        else:  # If it's a single line, add curly braces to the end and insert it at the beginning of the expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
    if add_bracket:
        res.extend(add_bracket)
    return res


def count_aoo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is not None and abc[1] is None and abc[2] is None
    return res


def convert_obo(node, code):
    # a for(;b;) c
    res, add_bracket = [], None
    abc = get_for_info_more_readable(node)
    if abc[0] is not None:  # If there is a
        indent = get_indent(node.start_byte, code)
        if abc[0].type != initMAP[get_lang()]:
            res.append((abc[0].end_byte, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
        else:
            res.append((abc[0].end_byte - 1, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
    if abc[2] is not None:  # If there is c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if abc[3] is None:
            return
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # The compound statement inserts if b break in the first sentence
            last_expression_node = abc[3].children[-2]
        else:  # If it is a single line, followed by curly braces, insert it at the beginning of expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
    if add_bracket:
        res.extend(add_bracket)
    return res


def count_obo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is None and abc[1] is not None and abc[2] is None
    return res


def convert_ooc(node, code):
    # a for(;;c) if b break
    res, add_bracket = [], None
    abc = get_for_info_more_readable(node)
    if abc[0] is not None:  # If there is a
        indent = get_indent(node.start_byte, code)
        if abc[0].type != initMAP[get_lang()]:
            res.append((abc[0].end_byte, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
        else:
            res.append((abc[0].end_byte - 1, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
    if abc[1] is not None:  # If there is b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # The compound statement inserts if b break in the first sentence
            first_expression_node = abc[3].children[1]
        else:  # If it is a single line, followed by curly braces, insert it at the beginning of expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        res.append(
            (
                first_expression_node.start_byte,
                f"if ({text(abc[1])})\n{(indent + 4) * ' '}break;\n{indent * ' '}",
            )
        )
    if add_bracket:
        res.extend(add_bracket)
    return res


def count_ooc(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is None and abc[1] is None and abc[2] is not None
    return res


def convert_ooo(node, code):
    # a for(;;;) if break b c
    res, add_bracket = [], None
    abc = get_for_info_more_readable(node)
    if abc is None:
        return
    if abc[0] is not None:  # If there is a
        indent = get_indent(node.start_byte, code)
        if abc[0].type != initMAP[get_lang()]:
            res.append((abc[0].end_byte, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
        else:
            res.append((abc[0].end_byte - 1, abc[0].start_byte))
            res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
    if abc[1] is not None:  # If there is b
        res.append((abc[1].end_byte, abc[1].start_byte))
        if abc[3] is None:
            return
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # The compound statement inserts if b break in the first sentence
            first_expression_node = abc[3].children[1]
        else:  # If it is a single line, followed by curly braces, insert it at the beginning of expression
            first_expression_node = abc[3]
        indent = get_indent(first_expression_node.start_byte, code)
        res.append(
            (
                first_expression_node.start_byte,
                f"if ({text(abc[1])})\n{(indent + 4) * ' '}break;\n{indent * ' '}",
            )
        )
    if abc[2] is not None:  # If there is c
        res.append((abc[2].end_byte, abc[2].start_byte))
        if (
            abc[3].type == blockMAP[get_lang()]
        ):  # The compound statement inserts if b break in the first sentence
            last_expression_node = abc[3].children[-2]
        else:  # If it is a single line, followed by curly braces, insert it at the beginning of expression
            last_expression_node = abc[3]
        indent = get_indent(last_expression_node.start_byte, code)
        res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
    if add_bracket:
        res.extend(add_bracket)
    return res


def count_ooo(node):
    nodes = match_for(node)
    res = 0
    for _node in nodes:
        abc = get_for_info_more_readable(_node)
        res += abc[0] is None and abc[1] is None and abc[2] is None
    return res
