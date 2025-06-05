import json
import numpy as np
import pandas as pd

def load_data_halu_eval(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_data_books(file_path):
    df = pd.read_excel(file_path)
    data = df.to_dict(orient='records')
    return data

def get_cosine_similarity(original_vec, vecs):
    original_vec = np.array(original_vec)
    vecs = np.array(vecs)

    dot_products = vecs.dot(original_vec)
    norm_original = np.linalg.norm(original_vec)
    norms_vecs = np.linalg.norm(vecs, axis=1)
    cosine_similarities = dot_products / (norms_vecs * norm_original)
    return np.mean(cosine_similarities)


