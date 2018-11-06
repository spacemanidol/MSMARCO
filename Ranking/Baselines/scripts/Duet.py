import torch
import torch.nn as nn
import torch.optim as optim

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