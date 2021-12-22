import numpy as np
import math
import torch

def test(data, model, pad_idx, device, args):
    model.eval()
    with torch.no_grad():
        all_correct_trials = []
        # losses = 0.0
        for iter, batch in enumerate(data):
            # transpose src and trg
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)

            # augment trg
            trg_input = trg[:, :-1]
            trg_out = trg[:, 1:]

            out, attn_wts = model(src, trg_input)

            preds = torch.argmax(out, axis=2)

            correct_pred = preds == trg_out
            correct_pred = correct_pred.cpu().numpy()
            mask = trg_out == pad_idx
            mask = mask.cpu().numpy()
            correct = np.logical_or(mask, correct_pred)
            correct = correct.all(0).tolist()
            all_correct_trials += correct
        accuracy = np.mean(all_correct_trials)
    model.train()
    return accuracy