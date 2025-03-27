import json
import os

def read_data(folder_path_halueval: str, folder_path_selfcheckgpt: str):
    """
        Return dataset_halu_eval, dataset_self_check_gpt
    """
    print("Load HaluEval 2.0")
    dataset_halu_eval = {}
    
    for file_name in os.listdir(folder_path_halueval):
        file_path = os.path.join(folder_path_halueval, file_name)
        file_name = file_name.replace(".json", "")
        
        with open(file_path, "r") as f:
            content = f.read()
        dataset_halu_eval[file_name] = json.loads(content)
        print(f"Length of {file_name}: {len(dataset_halu_eval[file_name])}.")
    
    print()
    print("Loading SelfCheckGPT")
    with open(folder_path_selfcheckgpt, "r") as f:
        content = f.read()
    dataset_self_check_gpt = json.loads(content)
    print(f"The length of the dataset: {len(dataset_self_check_gpt)}.")
    
    return dataset_halu_eval, dataset_self_check_gpt