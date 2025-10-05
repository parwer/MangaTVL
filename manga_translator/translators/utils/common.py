import re
import json
import yaml

def parse_response(content, debug=False):

    parsed = None
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    match2 = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)

    if debug:
        print("Raw response text:", content)  # Debugging line

    if match:
        json_str = match.group(1)
        parsed = yaml.safe_load(json_str)
    elif match2:
        json_str = match2.group(1)
        try:
            parsed = yaml.safe_load(json_str)
        except yaml.YAMLError:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as exc:
                print(content)
                raise TypeError("Cannot parse to JSON") from exc
    else:
        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            print(content)
            raise TypeError("Cannot parse to YAML") from exc

    if parsed is None:
        raise TypeError("Cannot parse to YAML due to NoneType")

    if isinstance(parsed, dict) and "panels" in parsed:
        parsed = parsed["panels"]

    return parsed