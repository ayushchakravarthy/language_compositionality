import numpy as np
import math
import torch

def test(data, model, pad_idx, device, args, loss_fn=False):
    model.eval()
    with torch.no_grad():
        all_correct_trials = []
        # losses = 0.0
        for iter, batch in enumerate(data):
            if args.model_type != "transformer_default":
                out, attn_wts = model(batch.src, batch.trg)
            else:
                out = model(batch.src, batch.trg)
            preds = torch.argmax(out, axis=2)
            # if loss_fn:
            #     print(batch.trg.T[15])
            #     print(preds.T[15])
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