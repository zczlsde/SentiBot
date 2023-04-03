from itertools import compress
import language_tool_python
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer, pipeline

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
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model.to(device)

    # Tokenize input texts
    texts = _read_text(path=path)
    rng = np.random.default_rng()
    if seed != None:
        rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(texts), size=size, replace=False)
    print("selecting indices: ", indices)
    texts = [texts[index] for index in indices]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

    # Calculate cosine similarities
    # results are in [0, 1]. Higher means more diversity
    cos_sim_mat = []
    for i in range(embeddings.shape[0]):
        for j in range(embeddings.shape[0]):
            if i >= j:
                continue
            cos_sim = cosine(embeddings[i], embeddings[j])/2
            cos_sim_mat.append(cos_sim)
    del model
    del inputs
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return cos_sim_mat

def novelty(training_phrase, path = "./Data/Generations/perc_generations.txt", seed=None, size = 50, start=None):
    """
    Novelty between {size} phrases and the trainning phrase
    training_phrase: the phrase used in the training dataset with the same prompt as the generated phrases
    path: the path of the generated txt file containing the phrases
    seed: for reproducibility
    start: select in order, seed is not used if start != None
    """
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model.to(device)
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
    inputs.to(device)
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

    # Calculate cosine similarities
    # results are in [0, 1]. Higher means more diversity
    cos_sim_mat = []
    for i in range(embeddings.shape[0]-1):
            cos_sim = cosine(embeddings[i], embeddings[-1])/2
            cos_sim_mat.append(cos_sim)
    del model
    del inputs
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return cos_sim_mat

def novelty_new(training_phrase, path = "./Data/Generations/perc_generations.txt", seed=None, size = 50, start=None, prompt_size=5):
    """
    Novelty between {size} phrases and the trainning phrase
    training_phrase: the phrase used in the training dataset with the same prompt as the generated phrases
    path: the path of the generated txt file containing the phrases
    seed: for reproducibility
    start: select in order, seed is not used if start != None
    """
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model.to(device)
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
    texts = [' '.join(text.split()[5:]) for text in texts]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

    # Calculate cosine similarities
    # results are in [0, 1]. Higher means more diversity
    cos_sim_mat = []
    for i in range(embeddings.shape[0]-1):
            cos_sim = cosine(embeddings[i], embeddings[-1])/2
            cos_sim_mat.append(cos_sim)

    del model
    del inputs
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return cos_sim_mat


def accuracy(path = "./Data/Generations/perc_generations.txt", size = None, metric=1):
    """
    Calculate the sentiment accuracy score for a generation file
    """
    phrases = _read_text(path)
    rng = np.random.default_rng()
    if size != None:
        indices = rng.choice(len(phrases), size)
        phrases = [phrases[index] for index in indices]
    print(len(phrases), " phrases")
    model = 'cardiffnlp/twitter-roberta-base-sentiment' if metric == 1 else 'nickwong64/bert-base-uncased-poems-sentiment'
    nlp = pipeline(task='text-classification', model=model)
    results = nlp(phrases)
    labels = [result['label'] for result in results]
    scores = [result['score'] for result in results]
    accuracy = [score if label == 'positive' or label == 'LABEL_2' else -score if label == 'negative' or label == 'LABEL_0' else 0 for label, score in zip(labels, scores)]
    return accuracy

def accuracy_smooth(path = "./Data/Generations/perc_generations.txt", size = None, metric=1):
    """
    Calculate the sentiment accuracy score for a generation file
    """
    device = torch.device("cuda")
    phrases = _read_text(path)
    rng = np.random.default_rng()
    if size != None:
        indices = rng.choice(len(phrases), size)
        phrases = [phrases[index] for index in indices]
    print(len(phrases), " phrases")
    model = 'cardiffnlp/twitter-roberta-base-sentiment' if metric == 1 else 'nickwong64/bert-base-uncased-poems-sentiment'
    nlp = pipeline(task='text-classification', model=model, top_k=None)
    results = nlp(phrases)
    scores = [labels['score'] for result in results for labels in result if labels['label'] == 'LABEL_2']
    neg_scores = [labels['score'] for result in results for labels in result if labels['label'] == 'LABEL_0']
    return scores, neg_scores


if __name__ == "__main__":
    # rate, rates = fluency(path="./Data/Generations/test.txt", size = 100)
    # div = diversity(path="./Data/Generations/test.txt", size=100)
    # sentiment, neg = accuracy_smooth()
    # print(sentiment)
    text = "To My Fairy Fancies NAY, no longer I may hold you, \
        In my spirit's soft caresses, Nor like lotus-leaves enfold you In the tangles of my tresses. \
            Fairy fancies, fly away To the white cloud-wildernesses, Fly away! Nay, no longer ye may linger With your laughter-lighted faces, \
                Now I am a thought-worn singer In life's high and lonely places. Fairy fancies, fly away, To bright wind-inwoven spaces, Fly"
    novelty_new(training_phrase=text)