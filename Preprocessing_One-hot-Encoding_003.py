# using keras on word-level one-hot encoding

from keras.preprocessing.text import Tokenizer
# import utilities for text-input processing

samples = ['this is a sample set of strings', 'And this is the other one.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
# builds the word index
# updates the set of vocabulary based on a list of texts

sequences = tokenizer.texts_to_matrix(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# converts a text to numpy matrix

word_index = tokenizer.word_index
