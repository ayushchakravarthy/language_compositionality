#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ucdavis
#SBATCH --mem=1G
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --output=logs/scan/transformer_def_mcd2.txt

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /home/akchak/.bashrc
source /home/tqhe/.bashrc
conda activate lp

python main.py \
--dataset scan \
--split mcd2 \
--num_runs 1 \
--batch_size 32 \
--num_epochs 100 \
--model_type transformer \
--d_model 12 \
--nhead 2 \
--n_layers 2 \
--dim_feedforward 20 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir transformer \
--out_data_file train_defaults_mcd2 \
--out_attn_wts train_defaults_mcd2_attn_maps \
--checkpoint_path ../weights/transformer/scan/defaults_mcd2.pt \
--checkpoint_every 4 \
--record_loss_every 20