import re
import string


def clean(inp: str) -> str:
    inp = inp.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    inp = re.sub(r"\s+", " ", inp.lower())
    return inp
