#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

conda init bash
conda activate lp

gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
for gpu in $gpus
do
echo "Setting fan for" $gpu "to full"
nvidia_fancontrol full $gpu
done

python main.py \
--split addjump \
--num_runs 10 \
--batch_size 32 \
--num_epochs 2 \
--model_type language_parser \
--d_model 12 \
--nhead 2 \
--ffn_exp 3 \
--patch_size 7 \
--num_enc_heads 2 \
--num_parts 16 \
--num_decoder_layers 2 \
--dim_feedforward 20 \
--dropout 0.1 \
--learning_rate 0.001 \
--results_dir language_parser \
--out_data_file train_defaults_jump \
--checkpoint_path ../weights/defaults_addjump.pt \
--checkpoint_every 1 \
--record_loss_every 20

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done