import re
import json
import yaml

def parse_response(content, debug=False):
    """Parse JSON or YAML from an LLM response (fenced or not)."""
    def try_parse_yaml(text):
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            if debug:
                print("YAML parsing failed:", exc)
            return None

    def try_parse_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            if debug:
                print("JSON parsing failed:", exc)
            return None

    def try_parse_first_json(text):
        # The model wrote a complete JSON value and then more ("Extra data": a stray "}",
        # a second object, a note) -> keep the first complete value.
        starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
        if not starts:
            return None
        try:
            return json.JSONDecoder().raw_decode(text[min(starts):])[0]
        except json.JSONDecodeError:
            return None

    if debug:
        print("Raw response text:\n", content)

    # Try extracting fenced code blocks
    match_json = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    match_generic = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)

    candidates = []
    if match_json:
        candidates.append(match_json.group(1))
    elif match_generic:
        candidates.append(match_generic.group(1))
    else:
        candidates.append(content)

    parsed = None
    for candidate in candidates:
        parsed = try_parse_yaml(candidate)
        if parsed is not None:
            break
        parsed = try_parse_json(candidate)
        if parsed is not None:
            break
        parsed = try_parse_first_json(candidate)
        if parsed is not None:
            break

    if parsed is None:
        print("Failed to parse response as JSON or YAML.")
        print(content)
        return []

    if not isinstance(parsed, (dict, list)):
        print("Parsed content is not a dict or list:")
        print(parsed)
        return []

    return parsed
