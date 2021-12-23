import re
import os
import numpy as np

def split_pos():
    l_s = [('prim', ['look', 'jump', 'walk', 'run'])]
    l_r = [('PRIM', ['I_LOOK', 'I_JUMP', 'I_WALK', 'I_RUN'])]

    for split in ['addjump', 'mcd1', 'mcd2', 'mcd3', 'simple']:
        for dataset in ['dev', 'test', 'train']:
            src = []
            trg = []
            with open(f'./{split}/{dataset}.src', 'r') as f:
                s = f.read()
                d = {k: "\\b(?:" + "|".join(v) + ")\\b" for k, v in l_s}
                for k, r in d.items(): s = re.sub(r, k, s)
                src.append(s)
            with open(f'./{split}/{dataset}.trg', 'r') as p:
                s = p.read()
                d = {k: "\\b(?:" + "|".join(v) + ")\\b" for k, v in l_r}
                for k, r in d.items(): s = re.sub(r, k, s)
                trg.append(s)
            print(f'{split}, {dataset} pos data processed')
            
            try:
                os.mkdir(f'./{split}/pos')
            except FileExistsError:
                print(f'{split}/pos: folder already created')

            
            with open(f'./{split}/pos/{dataset}.src', 'w') as q:
                for a in src:
                    q.write(a)
            with open(f'./{split}/pos/{dataset}.trg', 'w') as z:
                for b in trg:
                    z.write(b)
            print(f'{split}, {dataset} pos data written')

if __name__ == "__main__":
    split_pos()