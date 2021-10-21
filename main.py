import argparse
from train import train

parser = argparse.ArgumentParser()

# Training Data
parser.add_argument('--split',
                    choice = ['simple', 'addjump'],
                    help='SCAN split to use for training and testing')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs to do')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Samples per batch')
parser.add_argument('--num_epochs', type=int, default=2,
                    help='Number of training epochs')
