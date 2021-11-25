import argparse
from train import train
from plot import plot

parser = argparse.ArgumentParser()

# Training Data
parser.add_argument('--dataset', default='scan',
                    help='Dataset out of SCAN or COGS')
parser.add_argument('--split', default='simple',
                    help='SCAN split to use for training and testing')
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of runs to do')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Samples per batch')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of training epochs')

# Model
parser.add_argument('--model_type',
                    default='language_parser', help='Type of seq2seq model to use')
parser.add_argument('--d_model', type=int, default=12,
                    help="Dimension of inputs/outputs in transformer")
parser.add_argument('--nhead', type=int, default=2,
                    help='Number of heads in transformer with multihead attention')
parser.add_argument('--ffn_exp', type=int, default=3)
parser.add_argument('--num_parts', type=int, default=16)
parser.add_argument('--num_encoder_layers', type=int, default=2)
parser.add_argument('--num_decoder_layers', type=int, default=2)
parser.add_argument('--dim_feedforward', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--load_weights_from', default=None, required=False)

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001)

# Output options
parser.add_argument('--results_dir', default='language_parser',
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
    args = parser.parse_args()
    print(args)
    main(args)