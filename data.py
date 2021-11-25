# Utilities for SCAN dataset

import os
from sklearn.utils import shuffle

import torch
import torch.nn as nn

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import TranslationDataset

import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split


def build_scan(split, batch_size, attn_weights=False, device='cpu'):
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


    # Build Vocabulary
    SRC.build_vocab(train_)
    TRG.build_vocab(train_)

    if attn_weights:
        eval_path = os.path.join(path, 'eval')
        eval_ = TranslationDataset(eval_path, exts, fields)
        train, dev, test, eval = BucketIterator.splits((train_, dev_, test_, eval_),
                                             batch_size=batch_size,
                                             device=device)
        text = [
            f"run thrice after jump around left",
            f"jump right thrice and run left thrice", 
            f"jump around right twice after look left thrice", 
            f"jump thrice and look", 
            f"run opposite left after jump around right twice", 
            f"look opposite right after jump opposite left", 
            f"look opposite right thrice after jump", 
            f"look around left after jump opposite right", 
            f"jump opposite left twice after run right twice", 
            f"run opposite right after jump opposite left thrice", 
        ] 
        return SRC, TRG, train, dev, test, eval, text
    else:
        train, dev, test = BucketIterator.splits((train_, dev_, test_),
                                                 sort=False,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 device=device)
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