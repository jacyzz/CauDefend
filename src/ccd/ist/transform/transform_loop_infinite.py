# backdoor/attack/IST/transform/transform_loop_infinite.py

from ist_utils import text, get_indent
from transform.lang import get_lang

declaration_map = {"c": "declaration", "java": "local_variable_declaration"}
block_map = {"c": "compound_statement", "java": "block"}


def get_for_info(node):
    # Extract the ABC information of the for loop, for(a; b; c) and the following statement
    i, abc = 0, [None, None, None, None]
    for child in node.children:
        if child.type in [";", ")", declaration_map.get(get_lang())]:
            if child.type == declaration_map.get(get_lang()):
                abc[i] = child
            if child.prev_sibling and child.prev_sibling.type not in ["(", ";"]:
                abc[i] = child.prev_sibling
            i += 1
        if child.prev_sibling and child.prev_sibling.type == ")" and i == 3:
            abc[3] = child
    return abc


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


"""=========================match========================"""


def match_for_while(root):
    def check(node):
        if node.type in ["for_statement", "while_statement"]:
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


"""=========================replace========================"""


def count_inf_while(root):
    nodes = match_for_while(root)
    res = 0
    for node in nodes:
        if node.type == "for_statement":
            abc = get_for_info(node)
            if abc[1] and text(abc[1]) in ["1", "true"]:
                res += 1
        elif node.type == "while_statement":
            conditional_node = node.children[1].children[1]
            if text(conditional_node) in ["1", "true"]:
                res += 1
    return res


def cvt_infinite_while(node, code):
    # a while(b) c
    if node.type == "for_statement":
        res = []
        abc = get_for_info(node)
        
        block_node = None
        for c in node.children:
            if c.type == block_map.get(get_lang()):
                block_node = c
                break
        if not block_node: return []

        # Delete for(a; b; c)
        res.append((block_node.start_byte, node.start_byte))

        if abc[0] is not None:  # If there is a
            indent = get_indent(node.start_byte, code)
            a_text = text(abc[0])
            if abc[0].type != declaration_map.get(get_lang()):
                a_text += ';'
            res.append((node.start_byte, f'{a_text}\n{" " * indent}'))
            
        if abc[2] is not None and abc[3] is not None:  # If there is a c
            last_expression_node = block_node.children[-2] # before '}'
            indent = get_indent(last_expression_node.start_byte, code)
            res.append(
                (last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};")
            )
            
        while_str = f"while(true)"
        res.append((node.start_byte, while_str))

        if len(block_node.children) > 2 and abc[1] is not None: # has content and condition
            indent = get_indent(block_node.children[1].start_byte, code)
            break_str = f"\n{' '*indent}if(!({text(abc[1])})) break;"
            res.append((block_node.children[0].end_byte, break_str))

        return res
    
    elif node.type == "while_statement":
        # This part of the logic seems overly complex and might be buggy.
        # A simpler conversion would be to just change while(condition) to while(true)
        # and add if(!(condition)) break; at the start of the loop body.
        conditional_node = node.children[1].children[1]
        condition_str = text(conditional_node)
        
        block_node = node.children[-1]
        if len(block_node.children) <= 2: # empty block
            return []
            
        indent = get_indent(block_node.children[1].start_byte, code)
        
        res = []
        break_str = f"\n{' '*indent}if(!({condition_str})) break;"
        res.append((block_node.children[0].end_byte, break_str))

        res.append((conditional_node.end_byte, conditional_node.start_byte))
        res.append((conditional_node.start_byte, "true"))
        return res


# =================================================================
# ===== 在此处添加了 'infinite_for' 风格转换所需的新函数 =====
# =================================================================

def match_finite_for(root):
    """匹配可以被转换为无限循环的常规 for 循环。"""
    res = []
    
    def find_for_loops(node):
        if node.type == 'for_statement':
            # 简单检查：如果 for 循环有条件部分(b)，就认为它是有限的
            abc = get_for_info(node)
            if abc and abc[1] is not None:
                res.append(node)
        for child in node.children:
            find_for_loops(child)
            
    find_for_loops(root)
    return res

def convert_infinite_for(node, code):
    """将 for(a;b;c){...} 转换为 for(a;;){ if(!(b)) break; ... c; }"""
    block_node = None
    for c in node.children:
        if c.type == block_map.get(get_lang()):
            block_node = c
            break

    if not block_node or len(block_node.children) <= 2: # 无法转换没有代码块或空代码块的 for 循环
        return []

    abc = get_for_info(node)
    if not abc or abc[1] is None: # 已经是无限循环或格式错误
        return []

    a_text = text(abc[0]) if abc[0] else ""
    b_text = text(abc[1])
    c_text = text(abc[2]) if abc[2] else ""

    # 1. 创建新的 for 循环头部: for(a; ; )
    new_for_header = f"for({a_text}; ; )"
    
    # 2. 准备新的循环体内容
    # 获取循环体第一行代码的缩进
    first_stmt_in_block = block_node.children[1]
    indent = get_indent(first_stmt_in_block.start_byte, code)
    indent_str = ' ' * indent
    
    # 创建中断条件
    break_stmt = f"if(!({b_text})) break;"
    
    # 获取原始循环体内容 (不包括 '{' 和 '}')
    original_body_content = code[block_node.children[0].end_byte : block_node.children[-1].start_byte]
    
    # 将更新语句 'c' 添加到循环体末尾
    if c_text:
        # We need to find the correct indentation for 'c'
        last_stmt_in_block = block_node.children[-2]
        c_indent_str = ' ' * get_indent(last_stmt_in_block.start_byte, code)
        original_body_content += f"\n{c_indent_str}{c_text};"

    # 3. 组装新的 for 循环
    # 替换整个原始的 for 循环
    new_full_loop = f"{new_for_header} {{{break_stmt}{original_body_content}\n{' ' * get_indent(node.start_byte, code)}}}"
    
    return [(node.end_byte, node.start_byte), (node.start_byte, new_full_loop)]


def count_infinite_for(root):
    """计算 for(;;) 风格的无限循环数量。"""
    count = 0
    
    def find_inf_for(node):
        if node.type == 'for_statement':
            abc = get_for_info(node)
            # 无限循环没有条件部分 (b is None)
            if abc and abc[1] is None:
                count += 1
        for child in node.children:
            find_inf_for(child)
            
    find_inf_for(root)
    return count