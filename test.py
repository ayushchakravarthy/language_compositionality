import numpy as np
import operator
import torch

from itertools import compress

def test(data, model, pad_idx, device, args, save=False):
    model.eval()
    with torch.no_grad():
        all_correct_trials = []
        if args.dataset == 'scan':
            for i, batch in enumerate(data):
                # transpose src and trg
                src = batch['src']
                trg = batch['trg']
                try:
                    src_ann = batch['src_ann']
                    trg_ann_input = batch['trg_ann'][:, :-1]
                except:
                    src_ann = None
                    trg_ann_input = None

                # augment trg
                trg_input = trg[:, :-1]
                trg_out = trg[:, 1:]

                if args.model_type == 'sep-transformer':
                    out, adv_stat, attn_wts = model(src, trg_input, src_ann, trg_ann_input)
                else:
                    out, attn_wts = model(src, trg_input)

                preds = torch.argmax(out[0], axis=2)


                correct_pred = preds == trg_out
                correct_pred = correct_pred.cpu().numpy()
                mask = trg_out == pad_idx
                mask = mask.cpu().numpy()
                correct = np.logical_or(mask, correct_pred)
                correct = correct.all(1).tolist()
                if args.model_type == 'sep-transformer' and save:
                    wrong = list(map(operator.not_, correct))
                    w_idx = list(compress(range(len(wrong)), wrong))
                    enc_maps = []
                    dec_maps = []
                    for layer in range(args.n_layers):
                        if attn_wts['Encoder'] is not None:
                            enc_maps.append(attn_wts['Encoder'][layer][w_idx])

                        dec_maps.append({
                            'self': attn_wts['Decoder'][layer]['Sublayer1'][w_idx],
                            'mha': attn_wts['Decoder'][layer]['Sublayer2'][w_idx]
                        })
                    ret = (enc_maps, dec_maps, src[w_idx], trg_out[w_idx], preds[w_idx])
                else:
                    ret = None
                all_correct_trials += correct
        else:
            for batch in data:
                ret = None
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
                correct = correct.all(1).tolist()
                all_correct_trials += correct
        accuracy = np.mean(all_correct_trials)
    model.train()
    return accuracy, ret