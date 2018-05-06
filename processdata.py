from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from collections import Counter

import re
import numpy as np
import tensorflow as tf

import ttutils


def read_data():
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    f = open("few.txt", 'r')
    pattern = re.compile("[!,-.\"]")
    file = tf.compat.as_str(f.read()).lower()
    file = re.sub(pattern, '', file)
    words = file.split()
    return words


def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    ttutils.make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary


def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target


def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch


def process_data(vocab_size, batch_size, skip_window):
    words = read_data()
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words  # to save memory
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)


def get_index_vocab(vocab_size):
    words = read_data()
    return build_vocab(words, vocab_size)
