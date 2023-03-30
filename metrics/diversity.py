import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import the model
# Avaliable version for simcse are:
"""
    princeton-nlp/unsup-simcse-bert-base-uncased	76.25
    princeton-nlp/unsup-simcse-bert-large-uncased	78.41
    princeton-nlp/unsup-simcse-roberta-base	76.57
    princeton-nlp/unsup-simcse-roberta-large	78.90
    princeton-nlp/sup-simcse-bert-base-uncased	81.57
    princeton-nlp/sup-simcse-bert-large-uncased	82.21
    princeton-nlp/sup-simcse-roberta-base	82.52
    princeton-nlp/sup-simcse-roberta-large	83.76
"""
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

# Tokenize input texts
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output


# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))