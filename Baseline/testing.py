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
        lines = [line.strip() for line in f]

    phrases = []
    phrase = ""
    for x in lines:
        if x.startswith("==="):
            phrases.append(phrase)
            phrase = ""
            continue
        phrase = phrase + x
    return phrases

def fluency(path = "./Data/Generations/perc_generations.txt", verbose = 0, size=None):
    """
    Give a score on the average fluency and fluency distribution

    Returns:
            error rate: single float over all phrases
            error rates: array of float for each phrase
    """
    phrases = _read_text(path)
    rng = np.random.default_rng()
    if size != None:
        indices = rng.choice(len(phrases), size)
        phrases = [phrases[index] for index in indices]
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
            error_counter = 0
            for i in matches:
                if i.ruleId == "WHITESPACE_RULE" or i.ruleId == "PRP_COMMA" or i.ruleId == "COMMA_COMPOUND_SENTENCE":
                    continue
                error_counter += 1
            naive_errors = error_counter
            total_errors += naive_errors
            if length > 0:
                error_rates.append(1-naive_errors/length)
        except KeyboardInterrupt:
            break
    tool.close()
    error_rate = 1-total_errors/total_length
    return error_rate, error_rates


def diversity(path = "./Data/Generations/perc_generations.txt", seed=None, size = 50):
    """
    Diversity between {size} phrases
    path: the path of the generated txt file containing the phrases
    seed: for reproducibility
    """
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

    # Tokenize input texts
    texts = _read_text(path=path)
    rng = np.random.default_rng()
    if seed != None:
        rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(texts), size=size, replace=False)
    print("selecting indices: ", indices)
    texts = [texts[index] for index in indices]

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

def novelty(training_phrase, path = "./Data/Generations/perc_generations.txt", seed=None, size = 50, start=None):
    """
    Novelty between {size} phrases and the trainning phrase
    training_phrase: the phrase used in the training dataset with the same prompt as the generated phrases
    path: the path of the generated txt file containing the phrases
    seed: for reproducibility
    start: select in order, seed is not used if start != None
    """
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

    # Tokenize input texts
    texts = _read_text(path=path)
    if start == None:
        rng = np.random.default_rng()
        if seed != None:
            rng = np.random.default_rng(seed=seed)
        index = rng.choice(len(texts))
    else:
        index = start if start+size < len(texts) else 0
    texts = texts[index:index+size]
    texts.append(training_phrase)

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # results are in [0, 1]. Higher means more diversity
    cos_sim_mat = []
    for i in range(embeddings.shape[0]-1):
            cos_sim = cosine(embeddings[i], embeddings[-1])/2
            cos_sim_mat.append(cos_sim)
    return cos_sim_mat

def accuracy(path = "./Data/Generations/perc_generations.txt", seed=19019509, size = 50):
    return

if __name__ == "__main__":
    rate, rates = fluency(path="./Data/Generations/test.txt", size = 100)
    print(rate)
    div = diversity(path="./Data/Generations/test.txt", size=100)
    div2 = diversity(path="./Data/Generations/perc_generations.txt", size=100)
    print(np.mean(div), np.mean(div2))
