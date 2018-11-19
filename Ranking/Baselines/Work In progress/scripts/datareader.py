import csv
import re
import numpy as np

class DataReader:

    def __init__(self, data_file, num_meta_cols, multi_pass):
        self.num_meta_cols                  = num_meta_cols
        self.multi_pass                     = multi_passs
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
