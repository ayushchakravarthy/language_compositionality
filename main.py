import argparse
import torch
import numpy as np
import time 

import torch.multiprocessing as mp

random_seed = np.random.randint(1, 100000)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


from train import train

parser = argparse.ArgumentParser()

# Training Data
parser.add_argument('--dataset', default='scan',
                    help='Dataset out of SCAN, COGS, or PCFG Set')
parser.add_argument('--split', default='simple',
                    help='SCAN split to use for training and testing')
parser.add_argument('--pos', action='store_true',
                    help='use POS data for supervision')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs to do')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Samples per batch')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of training epochs')

# Model
# Transformer Arguments
parser.add_argument('--model_type',
                    default='language_parser', help='Type of seq2seq model to use')
parser.add_argument('--d_model', type=int, default=64,
                    help="Dimension of inputs/outputs in transformer")
parser.add_argument('--nhead', type=int, default=2,
                    help='Number of heads in transformer with multihead attention')
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--dim_feedforward', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--load_weights_from', default=None, required=False)

# TP-Separated Transformer Arguments
#TODO: rename cat_xm to something more relevant
parser.add_argument('--cat_xm', action='store_true',
                    help='concatenate X and M for output')
parser.add_argument('--sp_kernel', action='store_true',
                    help='use modified spherical gaussian kernel for similarity')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--encoding_scheme', default='absolute',
                    help='scheme for conveying positional information to the model')

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001)

# Output options
parser.add_argument('--results_dir', default='transformer',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='train_results.json',
                    help='Name of output data file with training loss data')
parser.add_argument('--out_attn_wts', default='attn_weights.pickle',
                    help='Name of output data file with attn weight maps in pickle file format')
parser.add_argument('--checkpoint_path',default=None,
                    help='Path to output saved weights.')
parser.add_argument('--checkpoint_every', type=int, default=4,
                    help='Epochs before evaluating model and saving weights')
parser.add_argument('--record_loss_every', type=int, default=20,
                    help='iters before printing and recording loss')

def main(args):
    for run in range(args.num_runs):
        train(run, args)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    s = time.time()
    args = parser.parse_args()
    print(args)
    main(args)
    e = time.time() - s
    print(e)