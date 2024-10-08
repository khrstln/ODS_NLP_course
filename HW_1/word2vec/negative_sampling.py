import numpy as np


def get_negative_samples(target, num_negative_samples, vocab_size):
    neg_samples = []
    while len(neg_samples) < num_negative_samples:
        neg_sample = np.random.randint(0, vocab_size)
        if neg_sample != target:
            neg_samples.append(neg_sample)
    return neg_samples
