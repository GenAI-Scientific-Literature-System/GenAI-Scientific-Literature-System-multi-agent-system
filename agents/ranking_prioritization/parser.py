import ast
import re


def parse_output(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        data = ast.literal_eval(match.group(0))
        if isinstance(data, list):
            return data
    except Exception:
        return []

    return []