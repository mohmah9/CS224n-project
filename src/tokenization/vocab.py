#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
vocab.py: Vocabulary Generation
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""
from sklearn.model_selection import train_test_split
from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
from utils import read_corpus, pad_sents
import sentencepiece as spm
import os
import numpy as np


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """
    def __init__(self, word2id=None):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1 # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry
    
    @staticmethod
    def from_subword_list(subword_list):
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """
    def __init__(self, src_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab

    @staticmethod
    def build(src_sents) -> 'Vocab':
        """ Build Vocabulary.
        @param src_sents (list[str]): Source subwords provided by SentencePiece
        @param tgt_sents (list[str]): Target subwords provided by SentencePiece
        """

        print('initialize source vocabulary ..')
        src = VocabEntry.from_subword_list(src_sents)

        return Vocab(src)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id), f, indent=2,ensure_ascii=False)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']

        return Vocab(VocabEntry(src_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words)' % (len(self.src))


def get_vocab_list(file_path, source, vocab_size):
    """ Use SentencePiece to tokenize and acquire list of unique subwords.
    @param file_path (str): file path to corpus
    @param source (str): tgt or src
    @param vocab_size: desired vocabulary size
    """ 
    spm.SentencePieceTrainer.train(input=file_path, model_prefix=source, vocab_size=vocab_size,unk_id=3)     # train the spm model
    sp = spm.SentencePieceProcessor()                                                               # create an instance; this saves .model and .vocab files 
    sp.load('{}.model'.format(source))                                                              # loads tgt.model or src.model
    sp_list = [sp.id_to_piece(piece_id) for piece_id in range(sp.get_piece_size())]                 # this is the list of subwords
    return sp_list 

def test_train_spliter():
    t =[]
    with open("../../data/CLEAN/CLEAN_UTF8_all.txt",'r', encoding="utf-8") as f:
        for i in f:
            t.append(i)
    for time in range(1,6):
        x_train, x_test = train_test_split(t, test_size=0.2, random_state=1000)
        with open("./datasets/text_train_"+str(time)+".txt",'w', encoding="utf-8") as f1:
            f1.writelines(x_train)
        with open("./datasets/text_test_"+str(time)+".txt",'w', encoding="utf-8") as f2:
            f2.writelines(x_test)

if __name__ == '__main__':
    # args = docopt(__doc__)
    vocab_size = [2000,3500,5000,6121]
    test_train_spliter()
    for vs in vocab_size:
        for i in range(1,6):
            if not os.path.exists("./datasets/model-"+str(vs)+"-"+str(i)):
                os.mkdir("./datasets/model-"+str(vs)+"-"+str(i))
            
            x_train_path = "./datasets/text_train_"+str(i)+".txt"
            x_test_path = "./datasets/text_test_"+str(i)+".txt"
            sents = get_vocab_list(x_train_path, source="./datasets/model-"+str(vs)+"-"+str(i)+"/"+"model-"+str(vs)+"-"+str(i), vocab_size=vs)
            vocab = Vocab.build(sents)
            vocab.save("./datasets/model-"+str(vs)+"-"+str(i)+"/vocab_file.json")
            test_words=[]
            with open("./datasets/text_test_"+str(i)+".txt",'r', encoding="utf-8") as f2:
                for data in f2:
                    for wd in data.split(" "):
                        test_words.append(wd)
            
            test_tokens=np.array(vocab.src.words2indices(test_words))
            unk_count = np.count_nonzero(test_tokens==3)
            with open("../../reports/tokenization.txt",'a') as f1:
                f1.write("With vocab size: "+str(vs)+"  and seed: "+str(i)+"  and test token count of: "+str(test_tokens.shape[0])+"  UNKNOWN precentage is: " +str(unk_count/test_tokens.shape[0]))
                f1.write("\n")
    
    vocab_size = 6121
    sents = get_vocab_list("../../data/CLEAN/CLEAN_UTF8_all.txt", source="../../model/tokenization/model-"+str(vocab_size), vocab_size=vocab_size)
    vocab = Vocab.build(sents)
    vocab.save("../../model/tokenization/vocab-file-"+str(vocab_size)+".json")
    
    # print('read in source sentences: %s' % args['--train-src'])
    # print('read in target sentences: %s' % args['--train-tgt'])

    # src_sents = get_vocab_list(args['--train-src'], source='src', vocab_size=21000)         
    # tgt_sents = get_vocab_list(args['--train-tgt'], source='tgt', vocab_size=8000)
    # vocab = Vocab.build(src_sents, tgt_sents)
    # print('generated vocabulary, source %d words, target %d words' % (len(src_sents), len(tgt_sents)))


    # vocab.save(args['VOCAB_FILE'])
    # print('vocabulary saved to %s' % args['VOCAB_FILE'])
