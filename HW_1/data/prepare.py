import re


def prepare_data_skip_gram(text: str, window_size=2):
    text = re.sub(r"[^a-z@# ]", "", text.lower())
    tokens = text.split()

    vocab = set(tokens)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    data = []
    for i in range(len(tokens)):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                data.append((tokens[i], tokens[j]))
    return data, word_to_ix, len(vocab)
