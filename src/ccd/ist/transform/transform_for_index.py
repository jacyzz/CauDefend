from ist_utils import text, find_descendants_by_type_name, find_son_by_type, find_son_by_name
from transform.lang import get_lang
from transform.transform_for_format import match_for, get_for_info_more_readable

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


"""==========================replacement========================"""

initDeclaratorMAP = {
    'java': 'variable_declarator',
    'c': 'init_declarator',
    'javascript': 'variable_declarator'
}

def convert_to_temp(node):
    abc = get_for_info_more_readable(node)
    init, cond, update, block = abc
    
    # * 没找到定义就不处理
    if init is None:
        return []

    res = []
    
    # * init
    if get_lang() in initDeclaratorMAP:
        index = find_son_by_type(init, initDeclaratorMAP[get_lang()]).children[0]
        target_index = 'temp'
        identifier_type = 'identifier'
    elif get_lang() == 'go':
        if len(init.children[0].children) == 0:
            # * for range 3
            return []
        index = init.children[0].children[0]
        target_index = 'temp'
        identifier_type = 'identifier'
    elif get_lang() == 'php':
        index = init.children[0]
        target_index = '$temp'
        identifier_type = 'variable_name'
        
    index_name = text(index)
    res.append((index.end_byte, index.start_byte))
    res.append((index.start_byte, target_index))
    
    # * cond
    if cond:
        index = find_son_by_name(cond, index_name)
        if index:
            res.append((index.end_byte, index.start_byte))
            res.append((index.start_byte, target_index))
    
    # * update
    if update:
        index = find_son_by_name(update, index_name)
        if index:
            res.append((index.end_byte, index.start_byte))
            res.append((index.start_byte, target_index))
    
    # * block
    indexs = find_descendants_by_type_name(block, identifier_type, index_name)
    for index in indexs:
        res.append((index.end_byte, index.start_byte))
        res.append((index.start_byte, target_index))
    
    return res


def count_for_index_temp(root):
    res = 0
    for node in match_for(root):
        abc = get_for_info_more_readable(node)
        init, cond, update, block = abc
        if get_lang() in initDeclaratorMAP:
            index = find_son_by_type(init, initDeclaratorMAP[get_lang()]).children[0]
            target_index = 'temp'
        elif get_lang() == 'go':
            if len(init.children[0].children) == 0:
                # * for range 3
                return []
            index = init.children[0].children[0]
            target_index = 'temp'
        elif get_lang() == 'php':
            index = init.children[0]
            target_index = '$temp'
        index_name = text(index)
        res += index_name == target_index
    return res