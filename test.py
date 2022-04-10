import numpy as np
import operator
import torch

from datasets import load_metric
from itertools import compress

def decode(pred, trg_vocab):
    predictions = []
    preds_txt = []
    for s in pred:
        prediction = ''
        for i in range(0, s.shape[0]):
            sym = trg_vocab.get_itos()[s[i]]
            if sym == '<eos>': break
            prediction += sym + ' '
        predictions.append(prediction.split())
        prediction += '<eos>'
        preds_txt.append(prediction)
    return predictions, preds_txt

def beam_search_decoder(log_posterior, k=10):
    """Beam Search Decoder

    Parameters:

        post(Tensor) – the posterior of network.
        k(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """
    batch_size, seq_length, _ = log_posterior.shape
    log_prob, indices = log_posterior[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_posterior[:, i, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)

    return indices, log_prob
    

def test(data, model, pad_idx, trg_vocab, args, save=False):
    model.eval()
    metric = load_metric("bleu")
    with torch.no_grad():
        all_correct_trials = []
        for _, batch in enumerate(data):
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
                out, attn_wts = model(src, trg_input, src_ann, trg_ann_input)
            else:
                out, attn_wts = model(src, trg_input, src_ann, trg_ann_input)

            
            if args.cat_xm:
                preds = torch.argmax(out[0], axis=2)
                beam_preds, log_prob = beam_search_decoder(out[0])
            else:
                preds = torch.argmax(out, axis=2)
                beam_preds, log_prob = beam_search_decoder(out)

            beam_translations, _ = decode(beam_preds[:, 0, :], trg_vocab)
            references, _ = decode(trg, trg_vocab)

            references = [[references[i][1:]] for i in range(len(references))]

            metric.add_batch(predictions=beam_translations, references=references)

            correct_pred = preds == trg_out
            correct_pred = correct_pred.cpu().numpy()
            mask = trg_out == pad_idx
            mask = mask.cpu().numpy()
            correct = np.logical_or(mask, correct_pred)
            correct = correct.all(1).tolist()
            all_correct_trials += correct
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
        accuracy = np.mean(all_correct_trials)
        final_bleu = metric.compute()
    model.train()
    return (accuracy, final_bleu["bleu"]), ret