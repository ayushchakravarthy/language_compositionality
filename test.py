import numpy as np
import torch

def test(data, model, pad_idx, device, args):
    model.eval()
    with torch.no_grad():
        all_correct_trials = []
        for batch in data:
            out, attn_wts = model(batch.src, batch.trg)
            preds = torch.argmax(out, dim=2)
            correct_pred = preds == batch.trg
            correct_pred = correct_pred.cpu().numpy()
            mask = batch.trg == pad_idx
            mask = mask.cpu().numpy()
            correct = np.logical_or(mask, correct_pred)
            correct = correct.all(0).tolist()
            all_collect_trials += correct
    accuracy = np.mean(all_correct_trials)
    model.train()
    return accuracy