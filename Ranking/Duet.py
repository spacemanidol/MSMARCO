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

DATA_DIR                        = '/home/erasmus/Documents/Ranking'
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
NUM_EPOCHS                      = 1
LEARNING_RATE                   = 1e-3
DATA_FILE_NGRAPHS               = os.path.join(DATA_DIR, "ngraphs.txt")
DATA_FILE_IDFS                  = os.path.join(DATA_DIR, "idf.norm.tsv")
DATA_FILE_TRAIN                 = os.path.join(DATA_DIR, "triples.train.small.tsv")
DATA_FILE_DEV                   = os.path.join(DATA_DIR, "top1000.dev.tsv")
QRELS_DEV                       = os.path.join(DATA_DIR, "qrels.dev.tsv")
DATA_FILE_NGRAPHS               = os.path.join(DATA_DIR, "ngraphs.txt")

def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

class DataReader:

    def __init__(self, data_file, num_meta_cols, multi_pass):
        self.num_meta_cols                  = num_meta_cols
        self.multi_pass                     = multi_pass
        self.regex_drop_char                = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space              = re.compile('\s+')
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
            self.features['dist_q']         = np.zeros((MB_SIZE, NUM_NGRAPHS, MAX_QUERY_TERMS), dtype=np.float32)
            self.features['dist_d']         = []
        for i in range(self.num_docs):
            if ARCH_TYPE != 1:
                self.features['local'].append(np.zeros((MB_SIZE, MAX_DOC_TERMS, MAX_QUERY_TERMS), dtype=np.float32))
            if ARCH_TYPE > 0:
                self.features['dist_d'].append(np.zeros((MB_SIZE, NUM_NGRAPHS, MAX_DOC_TERMS), dtype=np.float32))
        self.features['labels']             = np.zeros((MB_SIZE), dtype=np.int64)
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
        for i in range(MB_SIZE):
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
                                self.features['local'][d][i, j, k] = self.idfs[q[k]]
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
        for i in range(MB_SIZE):
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
                            self.features['local'][0][i, j, k] = self.idfs[q[k]]
                for j in range(len(prev_d)):
                    for k in range(len(q)):
                        if prev_d[j] == q[k]:
                            self.features['local'][1][i, j, k] = self.idfs[q[k]]
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
                        self.features['local'][1][0, j, k] = self.idfs[q[k]]
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
                                        nn.Dropout(p=DROPOUT_RATE))#,
                                        #nn.Linear(NUM_HIDDEN_NODES, 1),
                                        #nn.Tanh())
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
                                        nn.Dropout(p=DROPOUT_RATE))#,
                                        #nn.Linear(NUM_HIDDEN_NODES, 1),
                                        #nn.Tanh())
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

def main():
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

if __name__ == '__main__':
    main()