import spacy
import numpy as np
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util
from torchgen.gen_functionalization_type import convert_to_meta_tensors

nlp = spacy.load('en_core_web_sm')

def plot_histogram(lst, save_path, file_name, label='histogram'):
    bins = list(range(0, 101))
    plt.hist(lst, bins, label)
    plt.title(f'{file_name}')
    plt.xlabel('Intersection Ratio')
    plt.ylabel('Counts')
    plt.xticks(range(0, 101, 10))
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(f'{save_path}/{file_name}.png', dpi=300, bbox_inches='tight')
    plt.cla()

def gpt_embedding(gpt_model, text):
    response = gpt_model.embedding_request(text)
    response = response['data'][0]['embedding']
    return response

def gpt_query(gpt_model, query):
    response = gpt_model.submit_request(query)
    response = response['choices'][0]['text'].split("\n")
    return response

def similarity_sbert(text1, text2, embedding_model):
    embeddings1 = embedding_model.encode([text1], convert_to_tensor=True)
    embeddings2 = embedding_model.encode([text2], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()

def cosine_similarity_between_texts(text1, text2, embedding_model):
    embedding1 = embedding_model.submit_request(text1)
    embedding2 = embedding_model.submit_request(text2)
    embedding1_np = np.array(embedding1)
    embedding2_np = np.array(embedding2)
    cosine_similarity = np.dot(embedding1_np, embedding2_np) / (norm(embedding1_np) * norm(embedding2_np))
    return cosine_similarity
