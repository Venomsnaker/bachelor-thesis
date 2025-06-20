import json
import os
import numpy as np

def load_data_json(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_cosine_similarity(original_vec, vecs):
    original_vec = np.array(original_vec)
    vecs = np.array(vecs)

    dot_products = vecs.dot(original_vec)
    norm_original = np.linalg.norm(original_vec)
    norms_vecs = np.linalg.norm(vecs, axis=1)
    cosine_similarities = dot_products / (norms_vecs * norm_original)
    return np.mean(cosine_similarities)


