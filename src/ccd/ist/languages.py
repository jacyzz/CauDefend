from typing import Any
from tree_sitter import Parser, Language


# Map canonical language names to their tree-sitter module name.
# Ensure the corresponding python packages are available in the environment,
# e.g., tree_sitter_python, tree_sitter_java, etc.
SUPPORTED_LANGUAGES = {
    "python": "tree_sitter_python",
    "c": "tree_sitter_c",
    "javascript": "tree_sitter_javascript",
    "cpp": "tree_sitter_cpp",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
    "php": "tree_sitter_php",
}


def _load_language_module(language: str) -> Any:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    module_name = SUPPORTED_LANGUAGES[language]
    try:
        return __import__(module_name)
    except Exception as exc:
        # Fallback: try tree_sitter_languages (prebuilt grammars)
        try:
            from tree_sitter_languages import get_language  # type: ignore
            lang = get_language(language if language != "javascript" else "javascript")
            # Build a small shim exposing .language() or language_php() like the per-lang modules
            class _Shim:
                @staticmethod
                def language():
                    return lang
                @staticmethod
                def language_php():
                    return lang
            return _Shim()
        except Exception as exc2:
            raise ImportError(
                f"Failed to import tree-sitter module '{module_name}' "
                f"for language '{language}'. Please install the package or tree_sitter_languages."
            ) from exc2


def create_parser(language: str) -> Parser:
    """
    Create a tree-sitter Parser for the given language.
    The language module is expected to expose either `language()` or `language_<name>()`.
    """
    lang_module = _load_language_module(language)
    if language == "php":
        lang = Language(lang_module.language_php())
    else:
        lang = Language(lang_module.language())
    parser = Parser(lang)
    return parser


