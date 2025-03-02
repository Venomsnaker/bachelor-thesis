import numpy as np

class NLIConfig:
    nli_model: str = "potsawee/deberta-v3-large-mnli"

class LLMPromptConfig:
    model: str = "meta-llama/Llama-2-7b-chat-hf"

def expand_list1(my_list, num):
    expanded = []

    for x in my_list:
        for _ in range(num):
            expanded.append(x)
    return expanded

def expand_list2(my_list, num):
    expanded = []

    for _ in range(num):
        for x in my_list:
            expanded.append(x)
    return expanded

def smoothing(probs):
    probs = probs + 1e-12
    probs = probs / probs.sum()
    return probs

def kl_div(probs1, probs2):
    assert len(probs1) == len(probs2)
    probs1 = smoothing(probs1)
    probs2 =smoothing(probs2)
    xx = probs1 * np.log(probs1 / probs2)
    return xx.sum()

def onebest_argmax(probs1, probs2):
    answer1 = probs1.argmax()
    answer2 = probs2.argmax()

    if answer1 == answer2:
        count = 0
    else:
        count = 1
    return count

def hellinger_dist(probs1, probs2):
    sqrt_p1 = np.sqrt(probs1)
    sqrt_p2 = np.sqrt(probs2)
    return ((sqrt_p1 - sqrt_p2)**2).sum(axis=-1) / 1.4142135

def total_variation(probs1, probs2):
    diff = np.abs(probs1 - probs2)
    return diff.max()

def get_prob_distances(probs1, probs2):
    kl = kl_div(probs1, probs2)
    ob = onebest_argmax(probs1, probs2)
    hl = hellinger_dist(probs1, probs2)
    tv = total_variation(probs1, probs2)
    return kl, ob, hl, tv