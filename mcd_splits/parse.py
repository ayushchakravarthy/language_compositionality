import re
import json
import numpy as np

log_file = './tasks.txt'
regex_input = '(IN: .*)' 

index_file = './scan-splits/'

def parse():
    tasks = []
    with open(log_file, "r") as f:
        count = 0
        # Store src and trg strings into this list
        for line in f:
            for match in re.finditer(regex_input, line, re.S):
                match_text = match.group()
                string_parse = match_text.split('OUT: ')
                trg_string = string_parse[1]
                src_string = string_parse[0].split('IN: ')[1]
                tasks.append([src_string, trg_string])

    for split in ['mcd1', 'mcd2', 'mcd3']:
        with open(index_file + split + '.json', 'r') as f:
            file = json.load(f)
            train_idx = file.get('trainIdxs')
            dev_idx = file.get('devIdxs')
            test_idx = file.get('testIdxs')
            
            train_data = []
            dev_data = []
            test_data = []

            for i in train_idx:
                train_data.append(tasks[i])
            for j in dev_idx:
                dev_data.append(tasks[j])
            for k in test_idx:
                test_data.append(tasks[k])

            # store training data
            with open(f"../data/scan/{split}/train.src", 'w') as f:
                for t in train_data:
                    f.write(f"{t[0]}\n")
            with open(f"../data/scan/{split}/train.trg", 'w') as f:
                for e in train_data:
                    f.write(e[1])
            with open(f"../data/scan/{split}/test.src", 'w') as f:
                for text in test_data:
                    f.write(f"{text[0]}\n")
            with open(f"../data/scan/{split}/test.trg", 'w') as f:
                for text in test_data:
                    f.write(text[1])
            with open(f"../data/scan/{split}/dev.src", 'w') as f:
                for text in dev_data:
                    f.write(f"{text[0]}\n")
            with open(f"../data/scan/{split}/dev.trg", 'w') as f:
                for text in dev_data:
                    f.write(text[1])

if __name__ == "__main__":
    parse()
