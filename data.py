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

from torch.utils.data import Dataset

class PCFGSet(Dataset):
    """PCFG Set preprocessing"""

    def __init__(self, set, device='cpu', vocabs=None):
        self.device = device
        self.tokenizer = get_tokenizer("basic_english")

        path = 'data/pcfgset/pcfgset'

        assert set in ['train', 'dev', 'test']

        with open(f'{path}/{set}.src', 'r') as f:
            self.src = f.read().split('\n')
        with open(f'{path}/{set}.tgt', 'r') as f:
            self.trg = f.read().split('\n')

        self.src = self.tokenize(self.src[:-1])
        self.trg = self.tokenize(self.trg[:-1])

        if vocabs is None:
            self.SRC, self.TRG = self.build_vocab()
        else:
            (self.SRC, self.TRG) = vocabs

        self.src = self.to_int(self.src, self.SRC)
        self.trg = self.to_int(self.trg, self.TRG)

        self.src = self.pad(self.src)
        self.trg = self.pad(self.trg)

    def tokenize(self, str_list):
        s = []
        for string in str_list:
            s.append(self.tokenizer(string))
        return s

    def build_vocab(self):
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
        sample = {
            'src': self.src[index],
            'trg': self.trg[index]
        }

        return sample
        

class SCAN(Dataset):
    """SCAN dataset preprocessing"""
    
    def __init__(self, split, set, use_pos=False, device='cpu', vocabs=None):
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
        
        if vocabs is None:
            self.SRC, self.TRG = self.build_vocab()
        else:
            (self.SRC, self.TRG) = vocabs

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
            src_ann = self.src_pos[index]
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

class COGS(Dataset):
    """COGS preprocessing"""

    def __init__(self, set, split='train', device='cpu', vocabs=None):
        self.device = device
        self.tokenizer = get_tokenizer("basic_english")
        self.split = split

        path = 'data/cogs'

        assert set in ['train', 'dev', 'test', 'gen']
        if set != split:
            set = split

        with open(f'{path}/{set}.src', 'r') as f:
            self.src = f.read().split('\n')
        with open(f'{path}/{set}.tgt', 'r') as f:
            self.trg = f.read().split('\n')

        self.src = self.tokenize(self.src[:-1])
        self.trg = self.tokenize(self.trg[:-1])

        if vocabs is None:
            self.SRC, self.TRG = self.build_vocab()
        else:
            (self.SRC, self.TRG) = vocabs

        self.src = self.to_int(self.src, self.SRC)
        self.trg = self.to_int(self.trg, self.TRG)

        self.src = self.pad(self.src)
        self.trg = self.pad(self.trg)

    def tokenize(self, str_list):
        s = []
        for string in str_list:
            s.append(self.tokenizer(string))
        return s

    def build_vocab(self):
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
        sample = {
            'src': self.src[index],
            'trg': self.trg[index]
        }