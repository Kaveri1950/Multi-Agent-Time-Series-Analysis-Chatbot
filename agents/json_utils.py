"""Small local models often wrap requested JSON in extra prose/markdown
fences despite instructions. This pulls out the first {...} block and
parses it, so callers get a plain dict or a clear failure."""

import json
import re

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text!r}")
    return json.loads(match.group(0))
