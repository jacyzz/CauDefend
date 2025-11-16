"""
Style registry placeholder.

Paste/extend the full style mapping from your existing IST implementation.
Each entry maps a style code (e.g., "0.1") to a tuple:
    (style_type: str, style_subtype: str, prepare_styles: Optional[str])

The `prepare_styles` string can chain prerequisite styles, joined by '_' (e.g., "10.0_1.2").
"""
from typing import Dict, Tuple, Optional


StyleInfo = Tuple[str, str, Optional[str]]

# Initial subset as example. Replace/extend with your full style_dict.
STYLE_DICT: Dict[str, StyleInfo] = {
    "-3.1": ("tokensub", "sh", None),
    "-3.2": ("tokensub", "rb", None),
    "-2.1": ("invichar", "ZWSP", None),
    "-2.2": ("invichar", "ZWNJ", None),
    "-2.3": ("invichar", "LRO", None),
    "-2.4": ("invichar", "BKSP", None),
    "-1.1": ("deadcode", "deadcode_test_message", None),
    "-1.2": ("deadcode", "deadcode_233", None),
    "0.0": ("clean", "clean", None),
    "0.1": ("identifier_name", "camel", None),
    "0.2": ("identifier_name", "pascal", None),
    "0.3": ("identifier_name", "snake", None),
    "0.4": ("identifier_name", "hungarian", None),
    "0.5": ("identifier_name", "init_underscore", None),
    "0.6": ("identifier_name", "init_dollar", None),
    "1.1": ("bracket", "del_bracket", None),
    "1.2": ("bracket", "add_bracket", None),
    "2.1": ("augmented_assignment", "non_augmented", None),
    "2.2": ("augmented_assignment", "augmented", None),
    "3.1": ("cmp", "smaller", None),
    "3.2": ("cmp", "bigger", None),
    "3.3": ("cmp", "equal", None),
    "3.4": ("cmp", "not_equal", None),
    "4.1": ("for_update", "left", None),
    "4.2": ("for_update", "right", None),
    "4.3": ("for_update", "augment", None),
    "4.4": ("for_update", "assignment", None),
    "5.1": ("array_definition", "dyn_mem", None),
    "5.2": ("array_definition", "static_mem", None),
    "6.1": ("array_access", "pointer", None),
    "6.2": ("array_access", "array", None),
    "7.1": ("declare_lines", "split", None),
    "7.2": ("declare_lines", "merge", None),
    "8.1": ("declare_position", "first", None),
    "8.2": ("declare_position", "temp", None),
    "9.1": ("declare_assign", "split", None),
    "9.2": ("declare_assign", "merge", None),
    "10.0": ("for_format", "abc", None),
    "10.1": ("for_format", "obc", "10.0"),
    "10.2": ("for_format", "aoc", "10.0"),
    "10.3": ("for_format", "abo", "10.0"),
    "10.4": ("for_format", "aoo", "10.0"),
    "10.5": ("for_format", "obo", "10.0"),
    "10.6": ("for_format", "ooc", "10.0"),
    "10.7": ("for_format", "ooo", "10.0"),
    "11.1": ("for_while", "for", None),
    "11.2": ("for_while", "while", None),
    "11.3": ("for_while", "do_while", None),
    "11.4": ("loop_infinite", "infinite_while", None),
    "12.1": ("loop_infinite", "finite_for", None),
    "12.2": ("loop_infinite", "infinite_for", None),
    "12.3": ("loop_infinite", "finite_while", None),
    "12.4": ("loop_infinite", "infinite_while", None),
    "13.1": ("break_goto", "goto", None),
    "13.2": ("break_goto", "break", None),
    "14.1": ("if_exclamation", "not_exclamation", None),
    "14.2": ("if_exclamation", "exclamation", None),
    "15.1": ("if_return", "not_return", None),
    "15.2": ("if_return", "return", None),
    "16.1": ("if_switch", "switch", None),
    "16.2": ("if_switch", "if", None),
    "17.1": ("if_nested", "not_nested", None),
    "17.2": ("if_nested", "nested", None),
    "18.1": ("if_else", "not_else", None),
    "18.2": ("if_else", "else", None),
    "19.1": ("ternary", "to_ternary", None),
    "19.2": ("ternary", "to_if", None),
    "20.1": ("func_nested", "nested", None),
    "20.2": ("func_nested", "not_nested", None),
    "21.1": ("recursive_iterative", "to_iterative", None),
    "21.2": ("recursive_iterative", "to_recursive", None),
    "22.1": ("for_index", "temp", "10.0_1.2"),
}


