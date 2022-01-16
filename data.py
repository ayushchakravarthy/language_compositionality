# Utilities for SCAN dataset

from ast import Or
import os
from re import I

import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import TranslationDataset

from torch.utils.data import Dataset

class SCAN(Dataset):
    """SCAN dataset preprocessing"""
    
    def __init__(self, split, set, use_pos=False, device='cpu'):
        self.use_pos = use_pos
        self.device = device
        self.tokenizer = get_tokenizer("basic_english")

        # Get paths and filenames of each partition of split
        if split == 'simple':
            path = 'data/scan/simple'
        elif split == 'addjump':
            path = 'data/scan/addjump'
        elif split == 'mcd1':
            path = 'data/scan/mcd1'
        elif split == 'mcd2':
            path = 'data/scan/mcd2'
        elif split == 'mcd3':
            path = 'data/scan/mcd3'
        else:
            assert split not in ['simple', 'addjump', 'mcd1', 'mcd2', 'mcd3'], "Unknown split"
            exit()

        assert set in ['train', 'dev', 'test']

        with open(f'{path}/{set}.src', 'r') as f:
            self.src = f.read().split('\n')
        with open(f'{path}/{set}.trg', 'r') as f:
            self.trg = f.read().split('\n')

        self.src = self.tokenize(self.src[:-1])
        self.trg = self.tokenize(self.trg[:-1])

        if self.use_pos:
            with open(f'{path}/{set}.src.pos', 'r') as f:
                self.src_pos = f.read().split('\n')
            with open(f'{path}/{set}.trg.pos', 'r') as f:
                self.trg_pos = f.read().split('\n')
            self.src_pos = self.tokenize(self.src_pos[:-1])
            self.trg_pos = self.tokenize(self.trg_pos[:-1])
        
        self.SRC, self.TRG = self.build_vocab()

        self.src = self.to_int(self.src, self.SRC)
        self.trg = self.to_int(self.trg, self.TRG)

        if self.use_pos:
            self.src_pos = self.to_int(self.src_pos, self.SRC)
            self.trg_pos = self.to_int(self.trg_pos, self.TRG)

        self.src = self.pad(self.src)
        self.trg = self.pad(self.trg)

        if self.use_pos:
            self.src_pos = self.pad(self.src_pos)
            self.trg_pos = self.pad(self.trg_pos)

    def tokenize(self, str_list):
        s = []
        for string in str_list:
            s.append(self.tokenizer(string))
        return s

    def build_vocab(self):
        if self.use_pos:
            src = self.src + self.src_pos
            trg = self.trg + self.trg_pos
        else:
            src = self.src
            trg = self.trg

        SRC = build_vocab_from_iterator(src, specials=['<pad>', '<sos>', '<eos>'])
        TRG = build_vocab_from_iterator(trg, specials=['<pad>', '<sos>', '<eos>'])
        return SRC, TRG
    
    def to_int(self, str_list, vocab):
        l = []
        for s in str_list:
            l.append(torch.tensor([1] + vocab(s) + [2], device=self.device))
        return l

    def pad(self, str_list):
        return pad_sequence(str_list, batch_first=True, padding_value=0)
    
    def get_vocab(self):
        return self.SRC, self.TRG
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        if self.use_pos:
            src_ann = self.src_pos[index],
            trg_ann = self.trg_pos[index]
            sample = {
                'src': self.src[index],
                'trg': self.trg[index],
                'src_ann': src_ann,
                'trg_ann': trg_ann
            }
        else:
            sample = {
                'src': self.src[index],
                'trg': self.trg[index]
            }

        return sample
        
        
    

def build_scan(split, batch_size, use_pos, device):
    # Get paths and filenames of each partition of split
    if split == 'simple':
        path = 'data/scan/simple'
    elif split == 'addjump':
        path = 'data/scan/addjump'
    elif split == 'mcd1':
        path = 'data/scan/mcd1'
    elif split == 'mcd2':
        path = 'data/scan/mcd2'
    elif split == 'mcd3':
        path = 'data/scan/mcd3'
    else:
        assert split not in ['simple', 'addjump', 'mcd1', 'mcd2', 'mcd3'], "Unknown split"
        exit()

    train_path = os.path.join(path, 'train')
    dev_path = os.path.join(path, 'dev')
    test_path = os.path.join(path, 'test')

    exts = ('.src', '.trg')
    SRC = Field(init_token='<sos>', eos_token='<eos>')
    TRG = Field(init_token='<sos>', eos_token='<eos>')
    fields = (SRC, TRG)
    if use_pos:
        pos_exts = ('.src.pos', '.trg.pos')
        train_pos_ = TranslationDataset(train_path, pos_exts, fields)
        dev_pos_ = TranslationDataset(dev_path, pos_exts, fields)
        test_pos_ = TranslationDataset(test_path, pos_exts, fields)

        train_pos, dev_pos, test_pos = BucketIterator.splits((train_pos_, dev_pos_, test_pos_),
                              batch_size=batch_size,
                              sort=False,
                              shuffle=True,
                              device=device)
    else:
        train_pos, dev_pos, test_pos = [], [], []

    train_ = TranslationDataset(train_path, exts, fields)
    dev_ = TranslationDataset(dev_path, exts, fields)
    test_ = TranslationDataset(test_path, exts, fields)

    train_data, dev_data, test_data = BucketIterator.splits((train_, dev_, test_),
                                       sort=False,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       device=device)

    # Build Vocabulary
    if use_pos:
        SRC.build_vocab(train_, train_pos_)
        TRG.build_vocab(train_, train_pos_)
    else:
        SRC.build_vocab(train_)
        TRG.build_vocab(train_)

    return SRC, TRG, (train_data, dev_data, test_data), (train_pos, dev_pos, test_pos)

def build_cogs(batch_size, device='cpu'):
    path = 'data/cogs'
    train_path = os.path.join(path, 'train')
    dev_path = os.path.join(path, 'dev')
    test_path = os.path.join(path, 'test')
    train_100_path = os.path.join(path, 'train_100')
    gen_path = os.path.join(path, 'gen')


    exts = ('.src', '.trg')

    SRC = Field(init_token='<sos>', eos_token='<eos>')
    TRG = Field(init_token='<sos>', eos_token='<eos>')
    # this maybe needed later
    # dist = Field(init_token='<sos>', eos_token='<eos>')
    fields = (SRC, TRG)

    train_ = TranslationDataset(train_path, exts, fields)
    dev_ = TranslationDataset(dev_path, exts, fields)
    test_ = TranslationDataset(test_path, exts, fields)
    train_100_ = TranslationDataset(train_100_path, exts, fields)
    gen_ = TranslationDataset(gen_path, exts, fields)

    SRC.build_vocab(train_)
    TRG.build_vocab(train_)

    train, dev, test, train_100, gen = BucketIterator.splits((train_, dev_, test_, train_100_, gen_),
                                             sort=False,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             device=device)
    return SRC, TRG, train, train_100, dev, test, gen