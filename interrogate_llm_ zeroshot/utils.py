import json

def load_data_halu_eval(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data