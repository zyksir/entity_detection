#!/usr/bin/env python
#-*- coding: utf-8 -*-

# author: zyk

# input file:
# ../../data/subject_recognition/ + 'dev.txt'
#
# format:
# 'who/O was/O the/O trump/I ocean/I club/I international/I hotel/I and/I tower/I named/O after/O'
# 'where/O was/O sasha/I vujacic/I born/O'

import sys, os
import torch
from torch import nn
from torch.autograd import Variable
sys.path.append("../tools")
from utils import load_word2vec_format

input_path = '../../data/subject_recognition'


def load_pretrained_vectors(embed_file, word_dict, binary=False, normalize=True):
    assert embed_file is not None
    pretrained, vec_size, vocab = load_word2vec_format(embed_file, word_dict.word2index,
                                                       binary=binary, normalize=normalize)

    # Init Out-of-PreTrain Wordembedding using Min,Max Uniform
    scale = torch.std(pretrained)
    # random_range = (torch.min(pretrained), torch.max(pretrained))
    random_range = (-scale, scale)
    random_init_count = 0
    for word in word_dict:
        if word not in vocab:
            print(word)
            random_init_count += 1
            nn.init.uniform(pretrained[word_dict.lookup(word)], random_range[0], random_range[1])

    # Init 8901 words in uniform[tensor(-0.0534), tensor(0.0534)]
    print("Init %s words in uniform [%s, %s]" % (random_init_count, random_range[0], random_range[1]))
    return pretrained


class SubjectRecognitionLoader():
    def __init__(self, infile, device="cpu"):
        self.seqs, self.seq_lens, self.seq_labels = torch.load(infile)
        # print( type(self.seqs[0]) )
        self.batch_size = self.seqs[0].size(0)
        self.batch_num = len(self.seqs)

        if device != "cpu":
            for i in range(self.batch_num):
                self.seqs[i] = Variable(self.seqs[i].to(device))
                self.seq_labels[i] = Variable(self.seq_labels[i].to(device))

    def next_batch(self, shuffle=True):
        if shuffle:
            indices = torch.randperm(self.batch_num)
        else:
            indices = range(self.batch_num)
        for i in indices:
            yield self.seqs[i], self.seq_lens[i], self.seq_labels[i]