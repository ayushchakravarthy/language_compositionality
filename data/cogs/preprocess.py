import pandas as pd

def preprocess(split):
    data = pd.read_csv(f'raw/{split}.tsv', sep='\t')
    data.columns = ['eng', 'parse', 'dist']

    with open(f'{split}.src', 'w') as f:
        for x in data['eng']:
            f.write(f'{x}\n')
    with open(f'{split}.trg', 'w') as f:
        for y in data['parse']:
            f.write(f'{y}\n')


if __name__ == '__main__':
    preprocess('dev')
    preprocess('gen')
    preprocess('test')
    preprocess('train_100')
    preprocess('train')
