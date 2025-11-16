import os


def set_lang(lang):
    os.environ["code_language"] = lang


def get_lang():
    return os.environ["code_language"]
