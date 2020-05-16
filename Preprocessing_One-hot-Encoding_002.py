# character-level one-hot encoding

import string
import numpy as np

samples = ['this is a sample string of words',
           'this is another sample string of words']

# String of characters which are considered printable.
characters = string.printable
token_index = dict(zip(range(1, len(characters)+1), characters))

# Return a new dictionary initialized from an 
# optional positional argument and a possibly 
# empty set of keyword arguments.

max_lenght = 50
results = np.zeros((len(samples), max_length, max(token_index.keys())+1))

for i, sample in enumerate(samples):
    for j, characters in enumerate(sample):
        index = token_index.get(characters)
        results[i, j, index] = 1.
