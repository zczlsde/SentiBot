from itertools import compress
import language_tool_python
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

def _read_text(path):
    """
    read in the generated text and remove excess
    """
    with open(path, encoding="utf8") as f:
        lines = f.readlines()

    phrases = []
    phrase = ""
    for x in lines:
        if x.startswith("==="):
            phrases.append(phrase)
            phrase = ""
            continue
        phrase = phrase + x
    return phrases

def fluency(path = "./Data/Generations/perc_generations.txt", verbose = 0):
    """
    Give a score on the average fluency and fluency distribution

    Returns:
            error rate: single float over all phrases
            error rates: array of float for each phrase
    """
    phrases = _read_text(path)
    print(len(phrases), " phrases")

    # mask = [not x.startswith("===") for x in lines]
    # sentences = list(compress(lines, mask))   
    # check grammar error rate from the language tool
    tool = language_tool_python.LanguageTool("en-US")
    total_length = 0
    total_errors = 0
    error_rate = 0
    error_rates = []
    counter = 0
    for phrase in phrases:
        try:
            if verbose >= 1:
                counter += 1
                print(counter)
            length = len(phrase.split())
            total_length += length
            matches = tool.check(phrase)
            naive_errors = len(matches)
            total_errors += naive_errors
            if length > 0:
                error_rates.append(1-naive_errors/length)
        except KeyboardInterrupt:
            break
    tool.close()
    error_rate = 1-total_errors/total_length
    return error_rate, error_rates


def diversity(path = "./Data/Generations/perc_generations.txt", seed=19019509, size = 50):
    """
    Diversity between {size} phrases
    path: the path of the generated txt file containing the phrases
    seed: for reproducibility
    """
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

    # Tokenize input texts
    texts = _read_text(path=path)
    rng = np.random.default_rng(seed=seed)
    index = rng.choice(len(texts))
    texts = texts[index:index+size]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # results are in [0, 1]. Higher means more diversity
    cos_sim_mat = []
    for i in range(embeddings.shape[0]):
        for j in range(embeddings.shape[0]):
            if i >= j:
                continue
            cos_sim = cosine(embeddings[i], embeddings[j])/2
            cos_sim_mat.append(cos_sim)
    return cos_sim_mat

def novelty(training_phrase, path = "./Data/Generations/perc_generations.txt", seed=19019509, size = 50):
    """
    Novelty between {size} phrases and the trainning phrase
    training_phrase: the phrase used in the training dataset with the same prompt as the generated phrases
    path: the path of the generated txt file containing the phrases
    seed: for reproducibility
    """
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

    # Tokenize input texts
    texts = _read_text(path=path)
    rng = np.random.default_rng(seed=seed)
    index = rng.choice(len(texts))
    texts = texts[index:index+size]
    texts.append(training_phrase)

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # results are in [0, 1]. Higher means more diversity
    cos_sim_mat = []
    for i in range(embeddings.shape[0]):
            cos_sim = cosine(embeddings[i], embeddings[-1])/2
            cos_sim_mat.append(cos_sim)
    return cos_sim_mat



if __name__ == "__main__":
    # rate, rates = fluency(verbose=1)
    # print(rate, rates)
    div = diversity()
