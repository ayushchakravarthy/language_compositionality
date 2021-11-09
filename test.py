import numpy as np
import math
import torch

def test(data, model, pad_idx, device, args, loss_fn=None):
    model.eval()
    with torch.no_grad():
        all_correct_trials = []
        # losses = 0.0
        for batch in data:
            out, attn_wts = model(batch.src, batch.trg)
            # trg_out = batch.trg
            # loss = loss_fn(out.reshape(-1, out.shape[-1]), trg_out.reshape(-1))
            # losses += loss.item()
            preds = torch.argmax(out, axis=2)
            correct_pred = preds == batch.trg
            correct_pred = correct_pred.cpu().numpy()
            mask = batch.trg == pad_idx
            mask = mask.cpu().numpy()
            correct = np.logical_or(mask, correct_pred)
            correct = correct.all(0).tolist()
            all_correct_trials += correct
    accuracy = np.mean(all_correct_trials)
    model.train()
    return accuracy