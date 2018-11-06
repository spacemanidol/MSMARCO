#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
from __future__ import print_function
import json
import yaml
import argparse
import os
import sys
import csv
import re
import os.path
import itertools

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
import datetime
import h5py
import checkpointing

from scripts import DataReader
from scripts import Duet

import checkpointing
from dataset import load_data, tokenize_data, EpochGen
from dataset import SymbolEmbSourceNorm
from dataset import SymbolEmbSourceText
from dataset import symbol_injection


def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def try_to_resume(force_restart, exp_folder):
    if force_restart:
        return None, None, 0
    elif os.path.isfile(exp_folder + '/checkpoint'):
        checkpoint = h5py.File(exp_folder + '/checkpoint')
        epoch = checkpoint['training/epoch'][()] + 1
        # Try to load training state.
        try:
            training_state = torch.load(exp_folder + '/checkpoint.opt')
        except FileNotFoundError:
            training_state = None
    else:
        return None, None, 0

    return checkpoint, training_state, epoch


def reload_state(checkpoint, training_state, config, args):
    """
    Reload state when resuming training.
    """
    model, id_to_token, id_to_char = BidafModel.from_checkpoint(
        config['bidaf'], checkpoint)
    if torch.cuda.is_available() and args.cuda:
        model.cuda()
    model.train()

    optimizer = get_optimizer(model, config, training_state)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    with open(args.data) as f_o:
        data, _ = load_data(json.load(f_o),
                            span_only=True, answered_only=True)
    limit_passage = config.get('training', {}).get('limit')
    data = tokenize_data(data, token_to_id, char_to_id, limit_passage)

    data = get_loader(data, config)

    assert len(token_to_id) == len_tok_voc
    assert len(char_to_id) == len_char_voc

    return model, id_to_token, id_to_char, optimizer, data


def init_state(config, args):
    token_to_id = {'': 0}
    char_to_id = {'': 0}
    print('Loading data...')
    with open(args.data) as f_o:
        data, _ = load_data(json.load(f_o), span_only=True, answered_only=True)
    print('Tokenizing data...')
    data = tokenize_data(data, token_to_id, char_to_id)
    data = get_loader(data, config)

    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}

    print('Creating model...')
    model = BidafModel.from_config(config['bidaf'], id_to_token, id_to_char)

    if args.word_rep:
        print('Loading pre-trained embeddings...')
        with open(args.word_rep) as f_o:
            pre_trained = SymbolEmbSourceText(
                    f_o,
                    set(tok for id_, tok in id_to_token.items() if id_ != 0))
        mean, cov = pre_trained.get_norm_stats(args.use_covariance)
        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_token, 0,
                model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                pre_trained, oovs))
    else:
        pass  # No pretraining, just keep the random values.

    # Char embeddings are already random, so we don't need to update them.

    if torch.cuda.is_available() and args.cuda:
        model.cuda()
    model.train()

    optimizer = get_optimizer(model, config, state=None)
    return model, id_to_token, id_to_char, optimizer, data


def train(epoch, model, optimizer, data, args):
    """
    Train for one epoch.
    """

    for batch_id, (qids, passages, queries, answers, _) in enumerate(data):
        start_log_probs, end_log_probs = model(
            passages[:2], passages[2],
            queries[:2], queries[2])
        loss = model.get_loss(
            start_log_probs, end_log_probs,
            answers[:, 0], answers[:, 1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def main():
    """
    Main training program.
    """
    DEVICE                          = torch.device("cuda:0")    # torch.device("cpu"), if you want to run on CPU instead
ARCH_TYPE                       = 2
MAX_QUERY_TERMS                 = 20
MAX_DOC_TERMS                   = 200
NUM_HIDDEN_NODES                = 512
TERM_WINDOW_SIZE                = 3
POOLING_KERNEL_WIDTH_QUERY      = MAX_QUERY_TERMS - TERM_WINDOW_SIZE + 1 # 20 - 3 + 1 = 18
POOLING_KERNEL_WIDTH_DOC        = 100
NUM_POOLING_WINDOWS_DOC         = (MAX_DOC_TERMS - TERM_WINDOW_SIZE + 1) - POOLING_KERNEL_WIDTH_DOC + 1 # (200 - 3 + 1) - 100 + 1 = 99
NUM_NGRAPHS                     = 0
DROPOUT_RATE                    = 0.5
MB_SIZE                         = 1024 #1024
EPOCH_SIZE                      = 256 #1024
NUM_EPOCHS                      = 64
LEARNING_RATE                   = 1e-3
DATA_DIR                        = '/home/erasmus/Documents/Official Data/Ranking'
DATA_FILE_NGRAPHS               = os.path.join(DATA_DIR, "ngraphs.txt")
DATA_FILE_IDFS                  = os.path.join(DATA_DIR, "idf.norm.tsv")
DATA_FILE_TRAIN                 = os.path.join(DATA_DIR, "triples.train.small.tsv")
DATA_FILE_DEV                   = os.path.join(DATA_DIR, "top1000.dev.tsv")
QRELS_DEV                       = os.path.join(DATA_DIR, "qrels.dev.tsv")

READER_TRAIN                    = DataReader(DATA_FILE_TRAIN, 0, True)
READER_DEV                      = DataReader(DATA_FILE_DEV, 2, False)

qrels                           = {}
with open(QRELS_DEV, mode='r', encoding="utf-8") as f:
    reader                      = csv.reader(f, delimiter='\t')
    for row in reader:
        qid                     = int(row[0])
        did                     = int(row[2])
        if qid not in qrels:
            qrels[qid]          = []
        qrels[qid].append(did)

scores                          = {}
for qid in qrels.keys():
    scores[qid]                 = {}

torch.manual_seed(1)
print_message('Starting')
net                     = Duet()
net                     = net.to(DEVICE)
criterion               = nn.CrossEntropyLoss()
optimizer               = optim.Adam(net.parameters(), lr=LEARNING_RATE)
print_message('Number of learnable parameters: {}'.format(net.parameter_count()))
for ep_idx in range(NUM_EPOCHS):
    train_loss          = 0.0
    for docs in scores.values():
        docs.clear()
    net.train()
    for mb_idx in range(EPOCH_SIZE):
        features        = READER_TRAIN.get_minibatch()
        if ARCH_TYPE == 0:
            out         = torch.cat(tuple([net(torch.from_numpy(features['local'][i]).to(DEVICE), None, None) for i in range(READER_TRAIN.num_docs)]), 1)
        elif ARCH_TYPE == 1:
            out         = torch.cat(tuple([net(None, torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][i]).to(DEVICE)) for i in range(READER_TRAIN.num_docs)]), 1)
        else:
            out         = torch.cat(tuple([net(torch.from_numpy(features['local'][i]).to(DEVICE), torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][i]).to(DEVICE)) for i in range(READER_TRAIN.num_docs)]), 1)
        loss            = criterion(out, torch.from_numpy(features['labels']).to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss     += loss.item()
    is_complete         = False
    READER_DEV.reset()
    net.eval()
    while not is_complete:
        features        = READER_DEV.get_minibatch()
        if ARCH_TYPE == 0:
            out         = net(torch.from_numpy(features['local'][0]).to(DEVICE), None, None)
        elif ARCH_TYPE == 1:
            out         = net(None, torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][0]).to(DEVICE))
        else:
            out         = net(torch.from_numpy(features['local'][0]).to(DEVICE), torch.from_numpy(features['dist_q']).to(DEVICE), torch.from_numpy(features['dist_d'][0]).to(DEVICE))
        meta_cnt        = len(features['meta'])
        out             = out.data.cpu()
        for i in range(meta_cnt):
            q           = int(features['meta'][i][0])
            d           = int(features['meta'][i][1])
            scores[q][d]= out[i][0]
        is_complete     = (meta_cnt < MB_SIZE)
    mrr                 = 0
    for qid, docs in scores.items():
        ranked          = sorted(docs, key=docs.get, reverse=True)
        for i in range(len(ranked)):
            if ranked[i] in qrels[qid]:
                mrr    += 1 / (i + 1)
                break
    mrr                /= len(qrels)
    print_message('epoch:{}, loss: {}, mrr: {}'.format(ep_idx + 1, train_loss / EPOCH_SIZE, mrr))
print_message('Finished')

    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder")
    argparser.add_argument("data", help="Training data")
    argparser.add_argument("--force_restart",
                           action="store_true",
                           default=False,
                           help="Force restart of experiment: "
                           "will ignore checkpoints")
    argparser.add_argument("--word_rep",
                           help="Text file containing pre-trained "
                           "word representations.")
    argparser.add_argument("--cuda",
                           type=bool, default=torch.cuda.is_available(),
                           help="Use GPU if possible")
    argparser.add_argument("--use_covariance",
                           action="store_true",
                           default=False,
                           help="Do not assume diagonal covariance matrix "
                           "when generating random word representations.")

    args = argparser.parse_args()

    config_filepath = os.path.join(args.exp_folder, 'config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    checkpoint, training_state, epoch = try_to_resume(
            args.force_restart, args.exp_folder)

    if checkpoint:
        print('Resuming training...')
        model, id_to_token, id_to_char, optimizer, data = reload_state(
            checkpoint, training_state, config, args)
    else:
        print('Preparing to train...')
        model, id_to_token, id_to_char, optimizer, data = init_state(
            config, args)
        checkpoint = h5py.File(os.path.join(args.exp_folder, 'checkpoint'))
        checkpointing.save_vocab(checkpoint, 'vocab', id_to_token)
        checkpointing.save_vocab(checkpoint, 'c_vocab', id_to_char)

    if torch.cuda.is_available() and args.cuda:
        data.tensor_type = torch.cuda.LongTensor

    train_for_epochs = config.get('training', {}).get('epochs')
    if train_for_epochs is not None:
        epochs = range(epoch, train_for_epochs)
    else:
        epochs = itertools.count(epoch)

    for epoch in epochs:
        print('Starting epoch', epoch)
        train(epoch, model, optimizer, data, args)
        checkpointing.checkpoint(model, epoch, optimizer,
                                 checkpoint, args.exp_folder)

    return