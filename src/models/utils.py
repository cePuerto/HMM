import json
import torch as to

def load_json(jsonpath: str) -> dict:
    """Loads a JSON

    Args:
        jsonpath (str): JSON path

    Returns:
        dict: Dictionary with the JSON info
    """
    with open(jsonpath, "r", encoding="utf-8") as openfile:
        json_object = json.load(openfile)
    return json_object
