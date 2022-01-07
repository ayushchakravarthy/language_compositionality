# Utilities for SCAN dataset

import os

import torch
import torch.nn as nn
import numpy as np

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import TranslationDataset


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