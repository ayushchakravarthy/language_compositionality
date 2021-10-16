# Utilities for SCAN dataset

import os

# might make sense to try to use the huggingface stuff to build the dataset instead?
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import TranslationDataset


def build_scan(split, batch_size, device='cpu'):
    # Get paths and filenames of each partition of split
    if split == 'simple':
        path = 'data/scan/simple'
    elif split == 'addjump':
        path = 'data/scan/addjump'
    else:
        assert split not in ['simple', 'addjump'], "Unknown split"

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

    train, dev, test = BucketIterator.splits((train_, dev_, test_),
                                             batch_size=batch_size,
                                             device=device)
    return SRC, TRG, train, dev, test
