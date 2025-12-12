from ist_utils import replace_from_blob, traverse_rec_func, text
from transform.lang import get_lang

"""=========================match========================"""


def rec_LeftUpdate(node):
    # ++i or --i
    if node.type in ["update_expression"]:
        if node.parent.type not in [
            "subscript_expression",
            "argument_list",
            "assignment_expression",
        ]:
            # Not a[++i] Not *p(++i) Not a=++i
            if text(node.children[0]) in ["--", "++"]:
                return True
    return False


def rec_RightUpdate(node):
    # i++ or i--
    if node.type in ["update_expression"]:
        if node.parent.type not in [
            "subscript_expression",
            "argument_list",
            "assignment_expression",
        ]:
            # Not a[i++] Not *p(i++) Not a=i++
            if text(node.children[1]) in ["--", "++"]:
                return True


def rec_AugmentedCrement(node):
    # a += 1 or a -= 1
    if get_lang() == "python":
        # Python: augmented_assignment
        if node.type == "augmented_assignment":
            # children: [target, operator, value]
            if node.child_count >= 3 and text(node.children[1]) in ["+=", "-="]:
                return True
    else:
        if node.type == "assignment_expression":
            if (
                node.child_count >= 2
                and text(node.children[1]) in ["+=", "-="]
            ):
                return True


def rec_Assignment(node):
    # a = a ? 1
    if get_lang() == "python":
        if node.type == "assignment":
            # Accept "x = x + expr" or "x = expr + x" and "x = x - expr"
            def unwrap_paren(n):
                # unwrap parenthesized_expression recursively
                cur = n
                while cur is not None and cur.type == "parenthesized_expression" and cur.child_count >= 2:
                    # usually "(" expr ")"
                    cur = cur.children[1]
                return cur
            try:
                left_name = text(node.children[0]).strip()
                right = unwrap_paren(node.children[-1])
                if right is None or right.child_count < 3:
                    return False
                a = unwrap_paren(right.children[0])
                op = text(right.children[1]).strip()
                b = unwrap_paren(right.children[2])
                a_txt = text(a).strip() if a is not None else ""
                b_txt = text(b).strip() if b is not None else ""
                if op == "+":
                    # x = x + expr OR x = expr + x
                    if a_txt == left_name and b is not None:
                        return True
                    if b_txt == left_name and a is not None:
                        return True
                    return False
                elif op == "-":
                    # x = x - expr only
                    if a_txt == left_name and b is not None:
                        return True
                    return False
                else:
                    return False
            except Exception:
                return False
        return False
    else:
        if node.type == "assignment_expression":
            left_param = node.children[0].text
            if node.children[2].children:
                right_first_param = node.children[2].children[0].text
                if len(node.children[2].children) > 2:
                    if (
                        text(node.children[2].children[1]) in ["+", "-"]
                    ):
                        return left_param == right_first_param


def match_not_left(root):
    res = []

    def check(u):
        return rec_RightUpdate(u) or rec_AugmentedCrement(u) or rec_Assignment(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    return res


def match_not_right(root):
    res = []

    def check(u):
        return rec_LeftUpdate(u) or rec_AugmentedCrement(u) or rec_Assignment(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


def match_not_augment(root):
    res = []

    def check(u):
        return rec_LeftUpdate(u) or rec_RightUpdate(u) or rec_Assignment(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


def match_not_assignment(root):
    res = []

    def check(u):
        return rec_LeftUpdate(u) or rec_RightUpdate(u) or rec_AugmentedCrement(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


"""==========================replace========================"""


def convert_right(node):
    # i++
    if rec_LeftUpdate(node):
        temp_node = node.children[0]
        return [
            (temp_node.end_byte, temp_node.start_byte),
            (node.end_byte, text(temp_node)),
        ]
    if rec_AugmentedCrement(node):
        temp_node = node.children[0]
        op = text(node.children[1])[0]
        return [(node.end_byte, temp_node.end_byte), (temp_node.end_byte, op * 2)]
    if rec_Assignment(node):
        left_param = text(node.children[0])
        op = text(node.children[2].children[1])
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{left_param}{op*2}"),
        ]


def count_right(root):
    res = []

    def check(u):
        return rec_RightUpdate(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return len(res)


def convert_left(node):
    # ++i
    if rec_RightUpdate(node):
        temp_node = node.children[1]
        return [
            (temp_node.end_byte, temp_node.start_byte),
            (node.start_byte, text(temp_node)),
        ]
    if rec_AugmentedCrement(node):
        temp_node = node.children[0]
        op = text(node.children[1])[0]
        return [(node.end_byte, temp_node.end_byte), (temp_node.start_byte, op * 2)]
    if rec_Assignment(node):
        left_param = text(node.children[0])
        op = text(node.children[2].children[1])
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{op*2}{left_param}"),
        ]


def count_left(root):
    res = []

    def check(u):
        return rec_LeftUpdate(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return len(res)


def convert_augment(node):
    # i += 1
    if get_lang() == "python":
        if rec_Assignment(node):
            # "x = x + y" -> "x += y" ; "x = y + x" -> "x += y" ; "x = x - y" -> "x -= y"
            try:
                def unwrap_paren(n):
                    while n is not None and n.type == "parenthesized_expression" and n.child_count >= 2:
                        n = n.children[1]
                    return n
                param = text(node.children[0]).strip()
                right = unwrap_paren(node.children[-1])
                a = unwrap_paren(right.children[0])
                op = text(right.children[1]).strip()
                b = unwrap_paren(right.children[2])
                if op == "+":
                    if text(a).strip() == param:
                        expr = text(b).strip()
                    else:
                        expr = text(a).strip()
                    new_str = f"{param} += {expr}"
                else:  # op == "-"
                    # only x = x - expr supported here
                    expr = text(b).strip()
                    new_str = f"{param} -= {expr}"
                return [(node.end_byte, node.start_byte), (node.start_byte, new_str)]
            except Exception:
                return []
        # Python has no ++/--, ignore others
        return []
    else:
        if rec_LeftUpdate(node):
            op = text(node.children[0])[0]
            param = text(node.children[1])
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{param} {op}= 1")]
        if rec_RightUpdate(node):
            op = text(node.children[1])[0]
            param = text(node.children[0])
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{param} {op}= 1")]
        if rec_Assignment(node):
            param = text(node.children[0])
            op = text(node.children[2].children[1])
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{param} {op}= 1")]


def count_augment(root):
    res = []

    def check(u):
        return rec_AugmentedCrement(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return len(res)


def convert_assignment(node):
    # i = i + 1
    if get_lang() == "python":
        if rec_AugmentedCrement(node):
            # "x += y" -> "x = x + y" ; "x -= y" -> "x = x - y"
            try:
                param = text(node.children[0]).strip()
                op_full = text(node.children[1]).strip()
                op = "+" if op_full == "+=" else "-"
                expr = text(node.children[2]).strip()
                new_str = f"{param} = {param} {op} {expr}"
                return [(node.end_byte, node.start_byte), (node.start_byte, new_str)]
            except Exception:
                return []
        return []
    else:
        if rec_LeftUpdate(node):
            op = text(node.children[0])[0]
            param = text(node.children[1])
            return [
                (node.end_byte, node.start_byte),
                (node.start_byte, f"{param} = {param} {op} 1"),
            ]
        if rec_RightUpdate(node):
            op = text(node.children[1])[0]
            param = text(node.children[0])
            return [
                (node.end_byte, node.start_byte),
                (node.start_byte, f"{param} = {param} {op} 1"),
            ]
        if rec_AugmentedCrement(node):
            param = text(node.children[0])
            op = text(node.children[1])[0]
            return [
                (node.end_byte, node.start_byte),
                (node.start_byte, f"{param} = {param} {op} 1"),
            ]


def count_assignment(root):
    res = []

    def check(u):
        return rec_Assignment(u)

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return len(res)
