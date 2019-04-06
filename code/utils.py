#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from builtins import range
import numpy as np
import torch
import os
# import psycopg2
# import psycopg2.extras
from Levenshtein import distance
from functools import lru_cache

REAL = np.float32
if sys.version_info[0] >= 3:
    unicode = str


FB2M_KG_PATH = '/home2/zyk/kbqa/freebase_data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt'
FB5M_KG_PATH = '/home2/zyk/kbqa/freebase_data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt'
FB2M_KG_TABLE = 'fb_two_kg'
FB5M_KG_TABLE = 'fb_five_kg'
FB2M_NAME_TABLE = 'fb_two_subject_name'


def get_connection():
    # Load .env file
    pass_ = {}

    # Get the path relative to the directory this file is in
    _directory_path = os.path.dirname(os.path.realpath(__file__))
    pass_path = os.path.join(_directory_path, './.pass')
    for line in open(pass_path):
        split = line.strip().split('=')
        pass_[split[0]] = split[1]

    # Connect
    return psycopg2.connect(
        dbname=pass_['DB_NAME'],
        port=pass_['DB_PORT'],
        user=pass_['DB_USER'],
        host=pass_['DB_HOST'],
        password=pass_['DB_PASS'])


@lru_cache(maxsize=65536)
def edit_token_distance(needle, haystack):
    """
    Calculates the fuzzy match of needle in haystack, using a modified version of the Levenshtein
    distance algorithm.
    `levenshtein_word_aligned_distance` only allows the operations delete, insert and substitute of
    words. The cost of substituting a word is Levenshtein between the two words.
    Returns:
        distance: int value concerning the word aligned edit distance
        start_index: the start index of the min distance alignment between needle & haystack
        end_index: the end index of the min distance alignment between needle & haystack
    """
    m, n = len(needle), len(haystack)

    if not n:
        return m

    row1 = [0] * (n + 1)
    row1_start_index = list(range(0, n + 1))
    for i in range(0, m):
        row2 = [row1[0] + len(needle[i])]  # insertion
        row2_start_index = [row1_start_index[0]]
        for j in range(0, n):
            paths = [
                (row1[j + 1] + len(needle[i]), row1_start_index[j + 1]),  # deletion
                (row1[j] + distance(needle[i], haystack[j]), row1_start_index[j]),  # substitution
                (row2[j] + len(haystack[j]), row2_start_index[j]),  # insertion
            ]
            min_cost, start_index = min(paths, key=lambda p: p[0])
            row2.append(min_cost)
            row2_start_index.append(start_index)
        row1 = row2
        row1_start_index = row2_start_index

    min_cost = min(row1)
    # NOTE: multiple minimum cost spans
    min_spans = [(cost, row1_start_index[end_index], end_index)
                 for end_index, cost in enumerate(row1) if cost == min_cost]
    # NOTE: pick the smallest span that is farthest to the left
    min_cost, start_index, end_index = max(min_spans, key=lambda s: (s[1] - s[2], s[1]))
    return min_cost, int(start_index), int(end_index)


def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode.
    :param text:
    :param encoding:
    :param errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def any2utf8(text, encoding='utf8', errors='strict'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


def aeq(*args):
    base = args[0]
    for a in args[1:]:
        assert a == base, str(args)


def load_word2vec_format(filename, word_idx, binary=False, normalize=False,
                         encoding='utf8', unicode_errors='ignore'):
    """
    refer to gensim
    load Word Embeddings
    If you trained the C model using non-utf8 encoding for words, specify that
    encoding in `encoding`.
    :param filename :
    :param word_idx :
    :param binary   : a boolean indicating whether the data is in binary word2vec format.
    :param normalize:
    :param encoding :
    :param unicode_errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    vocab = set()
    print("loading word embedding from %s" % filename)
    with open(filename, 'rb') as fin:
        #header = to_unicode(fin.readline(), encoding=encoding)
        #vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        vocab_size = 1917494
        vector_size = 300
        word_matrix = torch.zeros(len(word_idx), vector_size)

        def add_word(_word, _weights):
            if _word not in word_idx:
                return
            vocab.add(_word)
            word_matrix[word_idx[_word]] = _weights

        if binary:
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                weights = torch.from_numpy(np.fromstring(fin.read(binary_len), dtype=REAL))
                add_word(word, weights)
        else:
            for line_no, line in enumerate(fin):
                parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                word, weights = parts[0], list(map(float, parts[1:]))
                weights = torch.Tensor(weights)
                add_word(word, weights)
    if word_idx is not None:
        assert (len(word_idx), vector_size) == word_matrix.size()
    if normalize:
        # each row normalize to 1
        word_matrix = torch.renorm(word_matrix, 2, 0, 1)
    print("loaded %d words pre-trained from %s with %d" % (len(vocab), filename, vector_size))
    return word_matrix, vector_size, vocab


def clip_weight_norm(model, max_norm, norm_type=2, except_params=None):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    for name, param in model.named_parameters():
        if except_params is not None:
            for except_param in except_params:
                if except_param in name:
                    # print "Pass", name
                    pass

        if len(param.size()) == 2:

            if name == 'out.linear.weight':
                row_norm = torch.norm(param.data, norm_type, 1)
                desired_norm = torch.clamp(row_norm, 0, np.sqrt(max_norm))
                scale = desired_norm / (row_norm + 1e-7)
                param.data = scale[:, None] * param.data
                # print "Row Norm", torch.norm(param.data, norm_type, 1)
            else:
                col_norm = torch.norm(param.data, norm_type, 0)
                desired_norm = torch.clamp(col_norm, 0, np.sqrt(max_norm))
                scale = desired_norm / (col_norm + 1e-7)
                param.data *= scale
                # print "Col Norm", torch.norm(param.data, norm_type, 0)
