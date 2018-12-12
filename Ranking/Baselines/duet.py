from __future__ import print_function
import sys
import os
import os.path
import csv
import re
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import msmarco_eval


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
MB_SIZE                         = 128
EPOCH_SIZE                      = 128
NUM_EPOCHS                      = 10000
LEARNING_RATE                   = 1e-3
DATA_DIR                        = 'data'
DATA_FILE_NGRAPHS               = os.path.join(DATA_DIR, "ngraphs.txt")
DATA_FILE_IDFS                  = os.path.join(DATA_DIR, "idf.norm.tsv")
DATA_FILE_TRAIN                 = os.path.join(DATA_DIR, "triples.train.full.tsv")
DATA_FILE_PREDICT                = os.path.join(DATA_DIR, "top1000.dev.tiny.tsv")
#DATA_FILE_PREDICT                = os.path.join(DATA_DIR, "top1000.dev.tsv")
DATA_FILE_PREDICTOUT              = os.path.join(DATA_DIR, "ranked.dev.tsv")
QRELS                       = os.path.join(DATA_DIR, "qrels.dev.tsv")


#DATA_FILE_PREDICTOUT               = os.path.join(DATA_DIR, "ranked.eval.tsv")
#QRELS                      = os.path.join(DATA_DIR, "qrels.eval.tsv")
#DATA_FILE_PREDICT                   = os.path.join(DATA_DIR, "top1000.eval.tsv")

class DataReader:
    def __init__(self, data_file, num_meta_cols, multi_pass, mb_size):
        self.num_meta_cols                  = num_meta_cols
        self.multi_pass                     = multi_pass
        self.regex_drop_char                = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space              = re.compile('\s+')
        self.mb_size                        = mb_size
        self.__load_ngraphs()
        self.__load_idfs()
        self.__init_data(data_file)
        self.__allocate_minibatch()
    def __tokenize(self, s, max_terms):
        return self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', s.lower())).strip().split()[:max_terms]
    def __load_ngraphs(self):
        global NUM_NGRAPHS
        self.ngraphs                        = {}
        self.max_ngraph_len                 = 0
        with open(DATA_FILE_NGRAPHS, mode='r', encoding="utf-8") as f:
            reader                          = csv.reader(f, delimiter='\t')
            for row in reader:
                self.ngraphs[row[0]]        = int(row[1]) - 1
                self.max_ngraph_len         = max(self.max_ngraph_len, len(row[0]))
        NUM_NGRAPHS                         = len(self.ngraphs)        
    def __load_idfs(self):
        self.idfs                           = {}
        with open(DATA_FILE_IDFS, mode='r', encoding="utf-8") as f:
            reader                          = csv.reader(f, delimiter='\t')
            for row in reader:
                self.idfs[row[0]]           = float(row[1])
    def __init_data(self, file_name):
        self.reader                         = open(file_name, mode='r', encoding="utf-8")
        self.num_docs                       = len(self.reader.readline().split('\t')) - self.num_meta_cols - 1
        self.reader.seek(0)    
    def __allocate_minibatch(self):
        self.features                       = {}
        if ARCH_TYPE != 1:
            self.features['local']          = [] 
        if ARCH_TYPE > 0:
            self.features['dist_q']         = np.zeros((self.mb_size, NUM_NGRAPHS, MAX_QUERY_TERMS), dtype=np.float32)
            self.features['dist_d']         = []
        for i in range(self.num_docs):
            if ARCH_TYPE != 1:
                self.features['local'].append(np.zeros((self.mb_size, MAX_DOC_TERMS, MAX_QUERY_TERMS), dtype=np.float32))
            if ARCH_TYPE > 0:
                self.features['dist_d'].append(np.zeros((self.mb_size, NUM_NGRAPHS, MAX_DOC_TERMS), dtype=np.float32))
        self.features['labels']             = np.zeros((self.mb_size), dtype=np.int64)
        self.features['meta']               = []        
    def __clear_minibatch(self):
        if ARCH_TYPE > 0:
            self.features['dist_q'].fill(np.float32(0))
        for i in range(self.num_docs):
            if ARCH_TYPE != 1:
                self.features['local'][i].fill(np.float32(0))
            if ARCH_TYPE > 0:
                self.features['dist_d'][i].fill(np.float32(0))
        self.features['meta'].clear()
    def get_minibatch(self):
        self.__clear_minibatch()
        for i in range(self.mb_size):
            row                             = self.reader.readline()
            if row == '':
                if self.multi_pass:
                    self.reader.seek(0)
                    row                     = self.dataFile.readline()
                else:
                    break
            cols                            = row.split('\t')
            q                               = self.__tokenize(cols[self.num_meta_cols], MAX_QUERY_TERMS)
            ds                              = [self.__tokenize(cols[self.num_meta_cols + i + 1], MAX_DOC_TERMS) for i in range(self.num_docs)]
            if ARCH_TYPE != 1:
                for d in range(self.num_docs):
                    for j in range(len(ds[d])):
                        for k in range(len(q)):
                            if ds[d][j] == q[k]:
                                if q[k] in self.idfs:
                                    self.features['local'][d][i, j, k] = self.idfs[q[k]]
                                else:
                                    self.features['local'][d][i, j, k] = 0
            if ARCH_TYPE > 0:
                for j in range(self.num_docs + 1):
                    terms = q if j == 0 else ds[j - 1]
                    for t in range(len(terms)):
                        term = '#' + terms[t] + '#'
                        term_len = len(term)
                        for k in range(term_len):
                            for l in range(1, self.max_ngraph_len + 1):
                                if k + l < term_len:
                                    ngraph_idx = self.ngraphs.get(term[k : k + l])
                                    if ngraph_idx != None:
                                        if j == 0:
                                            self.features['dist_q'][i, ngraph_idx, t] += 1
                                        else:
                                            self.features['dist_d'][j - 1][i, ngraph_idx, t] += 1
            self.features['meta'].append(tuple(cols[:self.num_meta_cols]))
        return self.features
    def get_minibatch2(self):
        self.__clear_minibatch()
        prev_d = []
        first_q = []
        for i in range(self.mb_size):
            row                             = self.reader.readline()
            if row == '':
                if self.multi_pass:
                    self.reader.seek(0)
                    row                     = self.dataFile.readline()
                else:
                    break
            cols                            = row.split('\t')
            q                               = self.__tokenize(cols[self.num_meta_cols], MAX_QUERY_TERMS)
            d                               = self.__tokenize(cols[self.num_meta_cols + 1], MAX_DOC_TERMS)
            if i == 0:
                first_q = q
            if ARCH_TYPE != 1:
                for j in range(len(d)):
                    for k in range(len(q)):
                        if d[j] == q[k]:
                            if q[k] in self.idfs:
                                self.features['local'][0][i, j, k] = self.idfs[q[k]]
                            else:
                                self.features['local'][0][i, j, k] = 0
                for j in range(len(prev_d)):
                    for k in range(len(q)):
                        if prev_d[j] == q[k]:
                            if q[k] in self.idfs:
                                self.features['local'][1][i, j, k] = self.idfs[q[k]]
                            else:
                                self.features['local'][1][i, j, k] = 0
            if ARCH_TYPE > 0:
                for j in range(self.num_docs + 1):
                    terms = q if j == 0 else (d if j == 1 else prev_d)
                    for t in range(len(terms)):
                        term = '#' + terms[t] + '#'
                        term_len = len(term)
                        for k in range(term_len):
                            for l in range(1, self.max_ngraph_len + 1):
                                if k + l < term_len:
                                    ngraph_idx = self.ngraphs.get(term[k : k + l])
                                    if ngraph_idx != None:
                                        if j == 0:
                                            self.features['dist_q'][i, ngraph_idx, t] += 1
                                        else:
                                            self.features['dist_d'][j - 1][i, ngraph_idx, t] += 1
            prev_d = d
            self.features['meta'].append(tuple(cols[:self.num_meta_cols]))
        if ARCH_TYPE != 1:
            for j in range(len(prev_d)):
                for k in range(len(first_q)):
                    if prev_d[j] == first_q[k]:
                        if q[k] in self.idfs:
                            self.features['local'][1][0, j, k] = self.idfs[q[k]]
                        else:
                            self.features['local'][1][0, j, k] = 0
        if ARCH_TYPE > 0:
            terms = prev_d
            for t in range(len(terms)):
                term = '#' + terms[t] + '#'
                term_len = len(term)
                for k in range(term_len):
                    for l in range(1, self.max_ngraph_len + 1):
                        if k + l < term_len:
                            ngraph_idx = self.ngraphs.get(term[k : k + l])
                            if ngraph_idx != None:
                                    self.features['dist_d'][1][0, ngraph_idx, t] += 1
        return self.features
    def reset(self):
        self.reader.seek(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


class Duet(torch.nn.Module):
    def __init__(self):
        super(Duet, self).__init__()
        self.duet_local             = nn.Sequential(nn.Conv1d(MAX_DOC_TERMS, NUM_HIDDEN_NODES, kernel_size=1),
                                        nn.Tanh(),
                                        Flatten(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES * MAX_QUERY_TERMS, NUM_HIDDEN_NODES),
                                        nn.Tanh(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.Tanh(),
                                        nn.Dropout(p=DROPOUT_RATE))
        self.duet_dist_q            = nn.Sequential(nn.Conv1d(NUM_NGRAPHS, NUM_HIDDEN_NODES, kernel_size=3),
                                        nn.Tanh(),
                                        nn.MaxPool1d(POOLING_KERNEL_WIDTH_QUERY),
                                        Flatten(),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.Tanh())
        self.duet_dist_d            = nn.Sequential(nn.Conv1d(NUM_NGRAPHS, NUM_HIDDEN_NODES, kernel_size=3),
                                        nn.Tanh(),
                                        nn.MaxPool1d(POOLING_KERNEL_WIDTH_DOC, stride=1),
                                        nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=1),
                                        nn.Tanh())
        self.duet_dist              = nn.Sequential(Flatten(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES * NUM_POOLING_WINDOWS_DOC, NUM_HIDDEN_NODES),
                                        nn.Tanh(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.Tanh(),
                                        nn.Dropout(p=DROPOUT_RATE))
        self.duet_comb              = nn.Sequential(nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.Tanh(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.Tanh(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, 1),
                                        nn.Tanh())
    def forward(self, x_local, x_dist_q, x_dist_d):
        if ARCH_TYPE != 1:
            h_local                 = self.duet_local(x_local)
        if ARCH_TYPE > 0:
            h_dist_q                = self.duet_dist_q(x_dist_q)
            h_dist_d                = self.duet_dist_d(x_dist_d)
            h_dist                  = self.duet_dist(h_dist_q.unsqueeze(-1)*h_dist_d)
        y_score                     = self.duet_comb((h_local + h_dist) if ARCH_TYPE == 2 else (h_dist if ARCH_TYPE == 1 else h_local))
        return y_score
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


def train(READER_TRAIN, net, optimizer, criterion):
    net.train()
    train_loss          = 0.0
    for mb_idx in range(EPOCH_SIZE):
        features        = READER_TRAIN.get_minibatch()
        acc = []
        x = torch.from_numpy(features['labels']).to(DEVICE)
        x2 = Variable(torch.from_numpy(features['dist_q'])).to(DEVICE)
        for i in range(READER_TRAIN.num_docs):
            x1 = Variable(torch.from_numpy(features['local'][i])).to(DEVICE)
            x3 = Variable(torch.from_numpy(features['dist_d'][i])).to(DEVICE)
            y = net(x1,x2,x3).to(DEVICE)
            acc.append(y)
        out = torch.cat(tuple(acc),1)
        loss = criterion(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss     += loss.item()
        del y, x, x1,x2,x3,acc, out,loss, features
        torch.cuda.empty_cache()
    return net, train_loss


def evaluate(net, qrels, scores, to_rerank):
    MB_SIZE = 2560
    READER_PREDICT = DataReader(DATA_FILE_PREDICT, 2, False, MB_SIZE)
    for docs in scores.values():
        docs.clear()
    is_complete         = False
    READER_PREDICT.reset()
    net.eval()
    while not is_complete:
        features        = READER_PREDICT.get_minibatch()
        meta_cnt        = len(features['meta'])
        is_complete     = (meta_cnt < MB_SIZE)
        x1 = Variable(torch.from_numpy(features['local'][0])).to(DEVICE)
        x2 = Variable(torch.from_numpy(features['dist_q'])).to(DEVICE)
        x3 = Variable(torch.from_numpy(features['dist_d'][0])).to(DEVICE)
        out         = net(x1,x2,x3).to(DEVICE).data.cpu().numpy()
        for i in range(meta_cnt):
            q           = int(features['meta'][i][0])
            d           = int(features['meta'][i][1])
            if q in scores:
                scores[q][d]= out[i][0]
        del x1,x2,x3, out, q, d ,features
        torch.cuda.empty_cache()
    mrr                 = 0
    for qid, docs in scores.items():
        ranked          = sorted(docs, key=docs.get, reverse=True)
        if len(ranked) > 0:
            for i in range(len(ranked)):
                if ranked[i] in qrels[qid]:
                    mrr    += 1 / (i + 1)
                    break
    mrr                /= len(to_rerank)
    with open(DATA_FILE_PREDICTOUT,'w') as w:
        for qid in to_rerank:
            ranked = sorted(scores[qid], reverse=False)
            for i in range(0, min(len(ranked),10)):
                w.write('{}\t{}\t{}\n'.format(qid,ranked[i],i+1))
    return mrr


def load_file_to_rerank(filename, offset):
    target,scores = {}, {}
    with open(filename,'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            qid = int(row[0])
            pid = int(row[1+offset])
            if qid not in target:
                target[qid]          = []
            target[qid].append(pid)
            scores[qid] = {}
    return target,scores            


def main():
    READER_TRAIN  = DataReader(DATA_FILE_TRAIN, 0, True, 256)
    qrels, scores = load_file_to_rerank(QRELS, 1)
    _, to_rerank = load_file_to_rerank(DATA_FILE_PREDICT,0)
    torch.manual_seed(1)
    print_message('Starting')
    try:
        net = torch.load('tensors.duet').to(DEVICE)
        print_message("Previous Model Found and Loaded")
    except:
        print_message("No Previous Model Found. Creating new")
        net = Duet().to(DEVICE)
    criterion               = nn.CrossEntropyLoss()
    optimizer               = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    print_message('Number of learnable parameters: {}'.format(net.parameter_count()))
    print_message('Training for {} epochs'.format(NUM_EPOCHS))
    for ep_idx in range(NUM_EPOCHS):
        net, train_loss  = train(READER_TRAIN,net, optimizer,criterion)
        print_message('epoch:{}, loss: {}'.format(ep_idx + 1, train_loss / EPOCH_SIZE))
        mrr = evaluate(net,qrels,scores, to_rerank)
        print_message('MRR @1000:{}'.format(mrr))
        torch.cuda.empty_cache()
        torch.save(net,'tensors.duet')
    print_message('Done Training')
    print_message('Evaluating')
    metrics = msmarco_eval.compute_metrics_from_files(QRELS,DATA_FILE_PREDICTOUT)
    print_message('#####################')
    for metric in sorted(metrics):
        print_message('{}: {}'.format(metric, metrics[metric]))
    print_message('#####################')            

if __name__ == '__main__':
    main()
