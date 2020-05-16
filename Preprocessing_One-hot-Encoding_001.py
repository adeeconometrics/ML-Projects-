# word-level one-hot encoding

import numpy as np

samples = ['this is a sample string of words',
           'this is another sample string of words']

token_index = {}  # dictionary

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index)+1

max_lenght = 10
results = np.zeros(shape=(len(samples)),
                   max_lenght,
                   max(token_index.values())+1)

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_lenght]:
        index = token_index.get(word)
        results[i, j, index] = 1
