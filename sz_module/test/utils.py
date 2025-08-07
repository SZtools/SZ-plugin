import json
import os

def load_test_input(key):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, "test_input.json")

    # Load JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data[key]