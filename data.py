# Utilities for SCAN dataset

import string
import os
import json
import tarfile
import mmap
from tqdm import tqdm

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

        path = '../data/pcfgset/pcfgset'

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
        assert split in ['simple', 'addjump', 'mcd1', 'mcd2', 'mcd3'], "Unknown Split"
        path = f'../data/scan/{split}'
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
            print(vocabs)
            self.SRC, self.TRG = vocabs
            exit()

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

    def __init__(self, split, set, use_pos=False, device='cpu', vocabs=None):
        self.device = device
        self.use_pos = use_pos
        # TODO: this tokenizer might not be the best for the target domain
        self.tokenizer = get_tokenizer("basic_english")
        self.split = split

        path = '../data/cogs'

        assert split in ['train', 'train-100'], 'Unknown split'
        assert set in ['train', 'dev', 'test', 'gen']

        if set == 'train' and split == 'train-100':
            set = split

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
            try:
                l.append(torch.tensor([1] + vocab(s) + [2], device=self.device))
            except RuntimeError:
                for w in s:
                    try:
                        vocab.append_token(w)
                    except RuntimeError:
                        continue
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

class CFQ(Dataset):
    def __init__(self):
        self.in_sentences = None
        self.out_sentences = None
        self.cache_dir = '../data'
        self.URL = "https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz"

        self.build()

    def tokenize_punctuation(self, text):
        # From https://github.com/google-research/google-research/blob/master/cfq/preprocess.py
        text = map(lambda c: ' %s ' % c if c in string.punctuation else c, text)
        return ' '.join(''.join(text).split())
    
    def preprocess_sparql(self, query):
        # From https://github.com/google-research/google-research/blob/master/cfq/preprocess.py
        """Do various preprocessing on the SPARQL query."""
        # Tokenize braces.
        query = query.replace('count(*)', 'count ( * )')

        tokens = []
        for token in query.split():
            # Replace 'ns:' prefixes.
            if token.startswith('ns:'):
                token = token[3:]
            # Replace mid prefixes.
            if token.startswith('m.'):
                token = 'm_' + token[2:]
            tokens.append(token)

        return ' '.join(tokens).replace('\\n', ' ')

    def load_data(self, fname: str):
        # Split the JSON manually, otherwise it requires infinite RAM and is very slow.
        pin = "complexityMeasures".encode()
        offset = 1
        cnt = 0

        inputs = []
        outputs = []

        with open(fname, "r") as f:
            data = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            pbar = tqdm(total=len(data))
            pbar.update(offset)

            while True:
                pos = data.find(pin, offset+6)
                if pos < 0:
                    this = data[offset: len(data)-2]
                else:
                    this = data[offset: pos-5]
                    new_offset = pos - 4
                    pbar.update(new_offset - offset)
                    offset = new_offset
                d = json.loads(this.decode())
                inputs.append(self.tokenize_punctuation(d["questionPatternModEntities"]))
                outputs.append(self.preprocess_sparql(d["sparqlPatternModEntities"]))

                cnt += 1
                if pos < 0:
                    break

        return inputs, outputs
    
    def build(self):
        index_table = {}

        if not os.path.isdir(os.path.join(self.cache_dir, "cfq")):
            gzfile = os.path.join(self.cache_dir, os.path.basename(self.URL))
            if not os.path.isfile(gzfile):
                assert False, f"Please download {self.URL} and place it in the {os.path.abspath(self.cache_dir)} "\
                               "folder. Google login needed."

            with tarfile.open(gzfile, "r") as tf:
                tf.extractall(path=self.cache_dir)

        splitdir = os.path.join(self.cache_dir, "cfq", "splits")
        for f in os.listdir(splitdir):
            if not f.endswith(".json"):
                continue

            name = f[:-5].replace("_split", "")
            with open(os.path.join(splitdir, f), "r") as f:
                ind = json.loads(f.read())

            index_table[name] = {
                "train": ind["trainIdxs"],
                "val": ind["devIdxs"],
                "test": ind["testIdxs"]
            }

        self.in_sentences, self.out_sentences = self.load_data(os.path.join(self.cache_dir, "cfq/dataset.json"))
        assert len(self.in_sentences) == len(self.out_sentences)
        print(index_table)

if __name__ == "__main__":
    cfq = CFQ()