# Utilities for SCAN dataset

from ast import Or
import os
from re import I

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, Counter

from torchtext.vocab import vocab

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import TranslationDataset

from torch.utils.data import Dataset

class SCAN(Dataset):
    """SCAN dataset preprocessing"""
    
    def __init__(self, split, set, use_pos=False, device='cpu'):
        self.use_pos = use_pos
        self.device = device

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

        if self.use_pos:
            with open(f'{path}/{set}.src.pos', 'r') as f:
                self.src_pos = f.read().split('\n')
            with open(f'{path}/{set}.trg.pos', 'r') as f:
                self.trg_pos = f.read().split('\n')

        self.SRC, self.TRG = self.build_vocab()
        self.to_int()

    def fill(self):
        src_max = max(self.src, key=len)
        for i, s in enumerate(self.src):
            s_l = s.split()
            s_l = ['<sos>'] + s_l + ['<eos>']
            for _ in range(len(s_l) - 1, len(src_max.split()) + 2):
                s_l.append('<pad>')
            s_r = ''
            for w in s_l:
                s_r += w + ' '
            s_r = s_r[:-1]
            self.src[i] = s_r
        trg_max = max(self.trg, key=len)
        for i, t in enumerate(self.trg):
            t_l = t.split()
            t_l = ['<sos>'] + t_l + ['<eos>']
            for _ in range(len(t_l) - 1, len(trg_max.split()) + 2):
                t_l.append('<pad>')
            t_r = ''
            for w in t_l:
                t_r += w + ' '
            t_r = t_r[:-1]
            self.trg[i] = t_r
        
        if self.use_pos:
            src_p_max = max(self.src_pos, key=len)
            for i, s in enumerate(self.src_pos):
                s_l = s.split()
                s_l = ['<sos>'] + s_l + ['<eos>']
                for _ in range(len(s_l) - 1, len(src_p_max.split()) + 2):
                    s_l.append('<pad>')
                s_r = ''
                for w in s_l:
                    s_r += w + ' '
                s_r = s_r[:-1]
                self.src_pos[i] = s_r
            trg_p_max = max(self.trg_pos, key=len)
            for i, t in enumerate(self.trg_pos):
                t_l = t.split()
                t_l = ['<sos>'] + s_l + ['<eos>']
                for _ in range(len(t_l) - 1, len(trg_p_max.split()) + 2):
                    t_l.append('<pad>')
                t_r = ''
                for w in t_l:
                    t_r += w + ' '
                t_r = t_r[:-1]
                self.trg_pos[i] = t_r

    def build_vocab(self):
        self.fill()

        if self.use_pos:
            src = self.src + self.src_pos
            trg = self.trg + self.trg_pos
        else:
            src = self.src
            trg = self.trg

        src_words = []
        for s in src:
            src_words += s.split(' ')
        src_words = src_words[:-1]
        counter = Counter(src_words)
        src_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        src_sorted = OrderedDict(src_sorted)
        SRC = vocab(src_sorted)
        
        trg_words = []
        for t in trg:
            trg_words += t.split(' ')
        trg_words = trg_words[:-1]
        counter = Counter(trg_words)
        trg_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        trg_sorted = OrderedDict(trg_sorted)
        TRG = vocab(trg_sorted)

        return SRC, TRG

    def to_int(self):
        self.src = self.src[:-1]
        self.trg = self.trg[:-1]

        src = []
        for s in self.src:
            l = []
            for w in s.split():
                l.append(self.SRC.get_stoi()[w])
            src.append(l)
        self.src = torch.tensor(src, dtype=torch.float, device=self.device)

        trg = []
        for s in self.trg:
            l = []
            for w in s.split():
                l.append(self.TRG.get_stoi()[w])
            trg.append(l)
        self.trg = torch.tensor(trg, dtype=torch.float, device=self.device)

        if self.use_pos:
            self.src_pos = self.src_pos[:-1]
            self.trg_pos = self.trg_pos[:-1]
            src_p = []
            for s in self.src_pos:
                l = []
                for w in s.split():
                    l.append(self.SRC.get_stoi()[w])
                src_p.append(l)
            self.src_pos = torch.tensor(src_p, dtype=torch.float, device=self.device)

            trg_p = []
            for s in self.trg_pos:
                l = []
                for w in s.split():
                    l.append(self.TRG.get_stoi()[w])
                trg_p.append(l)
            self.trg_pos = torch.tensor(trg_p, dtype=torch.float, device=self.device)

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