"""Phonetic frequency summaries used by the paper figures."""

import numpy as np
from spellitout.it_phonetics import split_into_phonemes


def calc_phonesyll_freq(sentence_list):
    syllables = [p for sentence in sentence_list
                 for word in split_into_phonemes(sentence).split()
                 for p in word.split(".")]
    phones, counts = np.unique(syllables, return_counts=True)
    order = np.argsort(counts)[::-1]
    return phones[order], counts[order]


def calc_phone_freq(sentence_list):
    phonemes = [p for sentence in sentence_list
               for word in split_into_phonemes(sentence).split()
               for p in word.replace(".", "").replace(" ", "")]
    return np.unique(phonemes, return_counts=True)
