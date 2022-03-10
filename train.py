# Training script

import os
import json
import pickle
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from data import SCAN, PCFGSet, COGS
from models.tf_separate import build_tp_sep_transformer
from models.tf import Transformer
from test import test


def train(run, args):
    # CUDA 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    comp_supervision = args.cat_xm

    print(args)

    # Data 
    if args.dataset == 'scan':
        train_data = SCAN(args.split, 'train', args.pos, device, None)
        SRC, TRG = train_data.get_vocab()

        dev_data = SCAN(args.split, 'dev', args.pos, device, (SRC, TRG))
        test_data = SCAN(args.split, 'test', args.pos, device, (SRC, TRG))            

        train_data = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0)
        dev_data = DataLoader(dev_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)
        test_data = DataLoader(test_data,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=0)
        
        # vocab
        src_vocab_size = len(SRC.get_stoi())
        trg_vocab_size = len(TRG.get_stoi())
        pad_idx = SRC['<pad>']
        assert TRG['<pad>'] == pad_idx
    
    elif args.dataset == 'pcfg-set':
        train_data = PCFGSet('train', device, None)
        SRC, TRG = train_data.get_vocab()

        dev_data = PCFGSet('dev', device, (SRC, TRG))
        test_data = PCFGSet('test', device, (SRC, TRG))

        train_data = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0
                                )
        dev_data = DataLoader(dev_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0
                              )
        test_data = DataLoader(test_data,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=0
                               )

        # vocab
        src_vocab_size = len(SRC.get_stoi())
        trg_vocab_size = len(TRG.get_stoi())
        pad_idx = SRC['<pad>']
        assert TRG['<pad>'] == pad_idx

        assert args.model_type in ['transformer']


    elif args.dataset == 'cogs':
        train_data = COGS(args.split, 'train', args.pos, device, None)
        # TODO: build a common vocab and save using torch.save and load each time here instead of building vocab for each run
        SRC, TRG = train_data.get_vocab()
        dev_data = COGS(args.split, 'dev', args.pos, device, (SRC, TRG))
        test_data = COGS(args.split, 'test', args.pos, device, (SRC, TRG))            
        gen_data = COGS(args.split, 'gen', args.pos, device, (SRC, TRG))            

        train_data = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0)
        dev_data = DataLoader(dev_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)
        test_data = DataLoader(test_data,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=0)

        gen_data = DataLoader(gen_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)
        
        # vocab
        src_vocab_size = len(SRC.get_stoi())
        trg_vocab_size = len(TRG.get_stoi())
        pad_idx = SRC['<pad>']
        assert TRG['<pad>'] == pad_idx


    if args.model_type == 'sep-transformer':
        model = build_tp_sep_transformer(args, pad_idx, src_vocab_size)
    elif args.model_type == 'transformer':
        model = Transformer(
            src_vocab_size,
            trg_vocab_size,
            args.d_model,
            args.nhead,
            args.n_layers,
            args.dim_feedforward,
            args.dropout,
            pad_idx,
            device, 
            args.pos
        )
    else:
        assert args.model_type not in ['transformer', 'sep-transformer']

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
    best_test_acc = float('-inf')
    
    wandb_dict = {}

    # Training Loop
    for epoch in range(args.num_epochs):
        for iter, batch in enumerate(train_data):
            # transpose src and trg
            src = batch['src']
            trg = batch['trg']

            if args.pos:
                src_ann = batch['src_ann']
                trg_ann_input = batch['trg_ann'][:, :-1]
                trg_ann_output = batch['trg_ann'][:, 1:]
            else:
                src_ann = None
                trg_ann_input = None

            # augment trg
            trg_input = trg[:, :-1]
            trg_out = trg[:, 1:]

            # pass through model and get predictions
            if args.model_type == 'sep-transformer':
                out, attn_wts = model(src, trg_input, src_ann, trg_ann_input)
                trg_vocab_size = src_vocab_size
            else:
                out, attn_wts = model(src, trg_input, src_ann, trg_ann_input)

            if comp_supervision:
                loss = loss_fn(out[0].view(-1, trg_vocab_size), trg_out.reshape(-1)) + \
                loss_fn(out[1].view(-1, trg_vocab_size), trg_ann_output.reshape(-1))
            else:
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
                    'Loss: ', loss_datapoint,
                )
                loss_data.append(loss_datapoint)
                wandb_dict['loss'] = loss

        # Checkpoint
        if epoch % args.checkpoint_every == 0:
            # Checkpoint on train data
            print("Checking training accuracy...")
            train_acc, _ = test(train_data, model, pad_idx, device, args)
            print("Training accuracy is ", train_acc)
            train_accs.append(train_acc)
            wandb_dict['train_acc'] = train_acc

            # Checkpoint on development data
            print("Checking development accuracy...")
            dev_acc, _ = test(dev_data, model, pad_idx, device, args)
            print("Development accuracy is ", dev_acc)
            dev_accs.append(dev_acc)
            wandb_dict['dev_acc'] = dev_acc

            # Checkpoint on test data
            print("Checking test accuracy...")
            test_acc, ret = test(test_data, model, pad_idx, device, args)
            print("Test accuracy is ", test_acc)
            test_accs.append(test_acc)
            wandb_dict['test_acc'] = test_acc
        
            if args.dataset == 'cogs':
                # Checkpoint on test data
                print("Checking gen accuracy...")
                gen_acc, ret = test(gen_data, model, pad_idx, device, args, True)
                print("Gen accuracy is ", gen_acc)
                gen_accs.append(gen_acc)
                wandb_dict['gen_acc'] = gen_acc
        
        wandb.log(wandb_dict)

        # Write stats file
        if args.dataset == 'pcfg-set':
            results_path = '../results/%s/%s' % (args.results_dir, args.dataset)
        else:
            results_path = '../results/%s/%s/%s' % (args.results_dir, args.dataset, args.split)

        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        if args.dataset != 'cogs':
            stats = {'loss_data':loss_data,
                     'train_accs':train_accs,
                     'dev_accs':dev_accs,
                     'test_accs':test_accs}
        else:
            stats = {'loss_data':loss_data,
                     'train_accs':train_accs,
                     'dev_accs':dev_accs,
                     'test_accs':test_accs,
                     'gen_accs': gen_accs}
        results_fn = '%s/%s%d.json' % (results_path,args.out_data_file,run)
        attn_file = '%s/%s%d.pickle' % (results_path, args.out_attn_wts, run)
        with open(results_fn, 'w') as f:
            json.dump(stats, f)
        
        # Save model weights
        if run == 0:
            if args.dataset == 'cogs':
                test_acc = gen_acc
            if test_acc > best_test_acc: # use dev to decide to save
                print(f'Saving best model: {epoch}')
                best_test_acc = test_acc
                if args.checkpoint_path is not None:
                    torch.save(model.state_dict(),
                               args.checkpoint_path)
                # Write attn weights to pickle file
                with open(attn_file, 'wb') as f:
                    pickle.dump(ret, f)