# Training script

import os
import json
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from data import build_cogs, build_scan
from models.models import LanguageParser, Transformer
from models.tp_separate import build_tp_sep_transformer
from test import test


def train(run, args):
    # CUDA 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Data 
    if args.dataset == 'scan':
        if args.pos:
            SRC, TRG, train_data, dev_data, test_data, SRC_pos, TRG_pos, train_data_pos, dev_data_pos, test_data_pos = build_scan(
                args.split,
                args.batch_size,
                use_pos=args.pos,
                device=device
            )
            # pos vocab
            src_pos_vocab_size = len(SRC_pos.vocab.stoi)
            trg_pos_vocab_size = len(TRG_pos.vocab.stoi)
            pos_pad_idx = SRC_pos.vocab[SRC_pos.pad_token]
            assert TRG_pos.vocab[TRG_pos.pad_token] == pos_pad_idx
            assert SRC.vocab[SRC.pad_token] == pos_pad_idx
        else:
            SRC, TRG, train_data, dev_data, test_data = build_scan(
                args.split,
                args.batch_size,
                device=device
            )
    elif args.dataset == 'cogs':
        SRC, TRG, train_data, train_100_data, dev_data, test_data, gen_data =  build_cogs(
            args.batch_size,
            device=device
        )
    # vocab
    src_vocab_size = len(SRC.vocab.stoi)
    trg_vocab_size = len(TRG.vocab.stoi)
    pad_idx = SRC.vocab[SRC.pad_token]
    assert TRG.vocab[TRG.pad_token] == pad_idx


    # Model
    if args.model_type == "language_parser":
        model = LanguageParser(
            src_vocab_size,
            trg_vocab_size,
            args.d_model,
            args.nhead,
            args.n_layers,
            args.dim_feedforward,
            args.dropout,
            pad_idx,
            device
        )
    elif args.model_type == "transformer":
        model = Transformer(
            src_vocab_size,
            trg_vocab_size,
            args.d_model,
            args.nhead,
            args.n_layers,
            args.dim_feedforward,
            args.dropout,
            pad_idx,
            device
        )
    elif args.model_type == 'sep-transformer':
        assert args.pos == True
        model = build_tp_sep_transformer(args, pad_idx, src_vocab_size)
    else:
        assert args.model_type not in ['transformer', 'language_parser', 'sep-transformer']

    print(f"Model size: {sum(p.numel() for p in model.parameters())}")
    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    if run == 0:
        print(model)
    model = model.to(device)
    model.train()

    # Loss Function
    loss_fn = nn.NLLLoss(ignore_index=pad_idx)
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Setup things to record
    loss_data = []
    train_accs = []
    dev_accs = []
    test_accs = []
    if args.dataset == 'cogs':
        gen_accs = []
    best_dev_acc = float('-inf')

    # Training Loop
    for epoch in range(args.num_epochs):
        if args.dataset == 'cogs':
            if args.split == 'train':
                for iter, batch in enumerate(train_data):
                    # transpose src and trg
                    src = batch.src.transpose(0, 1)
                    trg = batch.trg.transpose(0, 1)

                    # augment trg
                    trg_input = trg[:, :-1]
                    trg_out = trg[:, 1:]

                    # get predictions
                    out, attn_wts = model(src, trg_input)

                    loss = loss_fn(out.view(-1, trg_vocab_size), trg_out.view(-1))
                    optimizer.zero_grad()
                    optimizer.backward(loss)
                    optimizer.step()
                    # Record Loss
                    if iter % args.record_loss_every == 0:
                        loss_datapoint = loss.data.item()
                        print(
                            'Run:', run,
                            'Epochs: ', epoch,
                            'Iter: ', iter,
                            'Loss: ', loss_datapoint
                        )
                        loss_data.append(loss_datapoint)

                # Checkpoint
                if epoch % args.checkpoint_every == 0:
                    # Checkpoint on train data
                    print("Checking training accuracy...")
                    train_acc = test(train_data, model, pad_idx, device, args)
                    print("Training accuracy is ", train_acc)
                    train_accs.append(train_acc)

                    # Checkpoint on development data
                    print("Checking development accuracy...")
                    dev_acc = test(dev_data, model, pad_idx, device, args)
                    print("Development accuracy is ", dev_acc)
                    dev_accs.append(dev_acc)

                    # Checkpoint on test data
                    print("Checking test accuracy...")
                    test_acc = test(test_data, model, pad_idx, device, args)
                    print("Test accuracy is ", test_acc)
                    test_accs.append(test_acc)

                    print('Checking generalization accuracy...')
                    gen_acc = test(gen_data, model, pad_idx, device, args)
                    print("Generalization accuracy is ", gen_acc)
                    gen_accs.append(gen_acc)

                # Write stats file
                results_path = '../results/%s/%s/%s' % (args.results_dir, args.dataset, args.split)
                if not os.path.isdir(results_path):
                    os.mkdir(results_path)
                stats = {'loss_data':loss_data,
                         'train_accs':train_accs,
                         'dev_accs':dev_accs,
                         'test_accs':test_accs}
                results_fn = '%s/%s%d.json' % (results_path,args.out_data_file,run)
                attn_file = '%s/%s%d.pickle' % (results_path, args.out_attn_wts, run)
                with open(results_fn, 'w') as f:
                    json.dump(stats, f)
        
                # Write attn weights to pickle file
                with open(attn_file, 'wb') as f:
                    pickle.dump(attn_wts, f)

                # Save model weights
                if run == 0: #first run only
                    if dev_acc > best_dev_acc: # use dev to decide to save
                        best_dev_acc = dev_acc
                        if args.checkpoint_path is not None:
                            torch.save(model.state_dict(),
                                       args.checkpoint_path)
            elif args.split == 'train-100':
                for iter, batch in enumerate(train_100_data):
                    # transpose src and trg
                    src = batch.src.transpose(0, 1)
                    trg = batch.trg.transpose(0, 1)

                    # augment trg
                    trg_input = trg[:, :-1]
                    trg_out = trg[:, 1:]

                    # get predictions
                    out, attn_wts = model(src, trg_input)

                    loss = loss_fn(out.view(-1, trg_vocab_size), trg_out.view(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Record Loss
                    if iter % args.record_loss_every == 0:
                        loss_datapoint = loss.data.item()
                        print(
                            'Run:', run,
                            'Epochs: ', epoch,
                            'Iter: ', iter,
                            'Loss: ', loss_datapoint
                        )
                        loss_data.append(loss_datapoint)

                # Checkpoint
                if epoch % args.checkpoint_every == 0:
                    # Checkpoint on train data
                    print("Checking training accuracy...")
                    train_acc = test(train_data, model, pad_idx, device, args)
                    print("Training accuracy is ", train_acc)
                    train_accs.append(train_acc)

                    # Checkpoint on development data
                    print("Checking development accuracy...")
                    dev_acc = test(dev_data, model, pad_idx, device, args)
                    print("Development accuracy is ", dev_acc)
                    dev_accs.append(dev_acc)

                    # Checkpoint on test data
                    print("Checking test accuracy...")
                    test_acc = test(test_data, model, pad_idx, device, args)
                    print("Test accuracy is ", test_acc)
                    test_accs.append(test_acc)

                    print('Checking generalization accuracy...')
                    gen_acc = test(gen_data, model, pad_idx, device, args)
                    print("Generalization accuracy is ", gen_acc)
                    gen_accs.append(gen_acc)


                # Write stats file
                results_path = '../results/%s/%s/%s' % (args.results_dir, args.dataset, args.split)
                if not os.path.isdir(results_path):
                    os.mkdir(results_path)
                stats = {'loss_data':loss_data,
                         'train_accs':train_accs,
                         'dev_accs':dev_accs,
                         'test_accs':test_accs}
                results_fn = '%s/%s%d.json' % (results_path,args.out_data_file,run)
                attn_file = '%s/%s%d.pickle' % (results_path, args.out_attn_wts, run)
                with open(results_fn, 'w') as f:
                    json.dump(stats, f)
        
                # Write attn weights to pickle file
                with open(attn_file, 'wb') as f:
                    pickle.dump(attn_wts, f)

                # Save model weights
                if run == 0: #first run only
                    if dev_acc > best_dev_acc: # use dev to decide to save
                        best_dev_acc = dev_acc
                        if args.checkpoint_path is not None:
                            torch.save(model.state_dict(),
                                       args.checkpoint_path)
        elif args.dataset == 'scan':
            if args.pos and args.model_type == 'sep-transformer':
                for ((iter, batch), (_, batch_pos)) in zip(enumerate(train_data), enumerate(train_data_pos)):
                    # transpose src and trg
                    src = batch.src.transpose(0, 1)
                    trg = batch.trg.transpose(0, 1)
                    src_ann = batch_pos.src.transpose(0, 1)
                    trg_ann = batch_pos.trg.transpose(0, 1)

                    # augment trg
                    trg_input = trg[:, :-1]
                    trg_ann_input = trg_ann[:, :-1]
                    trg_out = trg[:, 1:]
                    trg_ann_out = trg_ann[:, 1:] # not going to be used for supervision

                    out, adv_stat, attn_wts = model(src, trg_input, src_ann, trg_ann_input)
                    loss = loss_fn(out.view(-1, src_vocab_size), trg_out.reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Record Loss
                    if iter % args.record_loss_every == 0:
                        loss_datapoint = loss.data.item()
                        print(
                            'Run:', run,
                            'Epochs: ', epoch,
                            'Iter: ', iter,
                            'Loss: ', loss_datapoint,
                            'Adversary Loss', adv_stat
                        )
                        loss_data.append(loss_datapoint)

            else:
                for iter, batch in enumerate(train_data):
                    # transpose src and trg
                    src = batch.src.transpose(0, 1)
                    trg = batch.trg.transpose(0, 1)

                    # augment trg
                    trg_input = trg[:, :-1]
                    trg_out = trg[:, 1:]

                    # pass through model and get predictions
                    out, attn_wts = model(src, trg_input)
                    loss = loss_fn(out.view(-1, trg_vocab_size), trg_out.reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Record Loss
                    if iter % args.record_loss_every == 0:
                        loss_datapoint = loss.data.item()
                        print(
                            'Run:', run,
                            'Epochs: ', epoch,
                            'Iter: ', iter,
                            'Loss: ', loss_datapoint
                        )
                        loss_data.append(loss_datapoint)

            # Checkpoint
            if epoch % args.checkpoint_every == 0:
                # Checkpoint on train data
                print("Checking training accuracy...")
                if args.pos:
                    train_acc = test(zip(enumerate(train_data), enumerate(train_data_pos)), model, pad_idx, device, args)
                else:
                    train_acc = test(train_data, model, pad_idx, device, args)
                print("Training accuracy is ", train_acc)
                train_accs.append(train_acc)

                # Checkpoint on development data
                print("Checking development accuracy...")
                if args.pos:
                    dev_acc = test(zip(enumerate(dev_data), enumerate(dev_data_pos)), model, pad_idx, device, args)
                else:
                    dev_acc = test(dev_data, model, pad_idx, device, args)
                print("Development accuracy is ", dev_acc)
                dev_accs.append(dev_acc)

                # Checkpoint on test data
                print("Checking test accuracy...")
                if args.pos:
                    test_acc = test(zip(enumerate(test_data), enumerate(test_data_pos)), model, pad_idx, device, args)
                else:
                    test_acc = test(test_data, model, pad_idx, device, args)
                print("Test accuracy is ", test_acc)
                test_accs.append(test_acc)

            # Write stats file
            results_path = '../results/%s/%s/%s' % (args.results_dir, args.dataset, args.split)
            if not os.path.isdir(results_path):
                os.mkdir(results_path)
            stats = {'loss_data':loss_data,
                     'train_accs':train_accs,
                     'dev_accs':dev_accs,
                     'test_accs':test_accs}
            results_fn = '%s/%s%d.json' % (results_path,args.out_data_file,run)
            attn_file = '%s/%s%d.pickle' % (results_path, args.out_attn_wts, run)
            with open(results_fn, 'w') as f:
                json.dump(stats, f)
        
            # Write attn weights to pickle file
            with open(attn_file, 'wb') as f:
                pickle.dump(attn_wts, f)

            # Save model weights
            if run == 0: #first run only
                if dev_acc > best_dev_acc: # use dev to decide to save
                    best_dev_acc = dev_acc
                    if args.checkpoint_path is not None:
                        torch.save(model.state_dict(),
                                   args.checkpoint_path)
        else:
            assert args.dataset not in ['scan', 'cogs'], "Unknown split"