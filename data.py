# Utilities for SCAN dataset

import os

import torch
import torch.nn as nn
import numpy as np

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import TranslationDataset


def build_scan(split, batch_size, use_pos=False, device='cpu'):
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

    train_path = os.path.join(path, 'train')
    dev_path = os.path.join(path, 'dev')
    test_path = os.path.join(path, 'test')
    exts = ('.src', '.trg')

    SRC = Field(init_token='<sos>', eos_token='<eos>')
    TRG = Field(init_token='<sos>', eos_token='<eos>')
    fields = (SRC, TRG)

    train_ = TranslationDataset(train_path, exts, fields)
    dev_ = TranslationDataset(dev_path, exts, fields)
    test_ = TranslationDataset(test_path, exts, fields)

    train, dev = BucketIterator.splits((train_, dev_),
                                       sort=False,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       device=device)
    test = BucketIterator(test_,
                          batch_size=batch_size,
                          sort=False,
                          shuffle=True,
                          train=False,
                          device=device)

    # Build Vocabulary
    SRC.build_vocab(train_)
    TRG.build_vocab(train_)

    if use_pos:
        if split == 'simple':
            path = 'data/scan/simple/pos'
        elif split == 'addjump':
            path = 'data/scan/addjump/pos'
        elif split == 'mcd1':
            path = 'data/scan/mcd1/pos'
        elif split == 'mcd2':
            path = 'data/scan/mcd2/pos'
        elif split == 'mcd3':
            path = 'data/scan/mcd3/pos'
        else:
            assert split not in ['simple', 'addjump', 'mcd1', 'mcd2', 'mcd3'], "Unknown split"

        train_path = os.path.join(path, 'train')
        dev_path = os.path.join(path, 'dev')
        test_path = os.path.join(path, 'test')
        exts = ('.src', '.trg')

        SRC_pos = Field(init_token='<sos>', eos_token='<eos>')
        TRG_pos = Field(init_token='<sos>', eos_token='<eos>')
        fields = (SRC_pos, TRG_pos)

        train_ = TranslationDataset(train_path, exts, fields)
        dev_ = TranslationDataset(dev_path, exts, fields)
        test_ = TranslationDataset(test_path, exts, fields)

        train_pos, dev_pos = BucketIterator.splits((train_, dev_),
                                           sort=False,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           device=device)
        test_pos = BucketIterator(test_,
                              batch_size=batch_size,
                              sort=False,
                              shuffle=True,
                              train=False,
                              device=device)

        # Build Vocabulary
        SRC_pos.build_vocab(train_)
        TRG_pos.build_vocab(train_)

        return SRC, TRG, train, dev, test, SRC_pos, TRG_pos, train_pos, dev_pos, test_pos
    else:
        return SRC, TRG, train, dev, test

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