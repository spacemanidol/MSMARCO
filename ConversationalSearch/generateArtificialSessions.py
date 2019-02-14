import json
import math
import numpy as np
import sys
from annoy import AnnoyIndex

TREESIZE = 10000
EMBEDDINGDIM = 100

def loadQueries(filename):
    queries = set()
    with open(filename,'r') as f:
        for l in f:
            queries.add(l.strip())
    return queries
def loadSessions(filename):
    sessions = []
    with open(filename,'r'):
        for l in f:
            sessions.append(l.strip().split('\t'))
    return sessions
def loadVectors(filename, realQueries):
    i = 0
    j = 0
    artificial = [{},{},[]]
    real = [{},{},[]]
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            query = l[0]
            vectors = l[1].split(' ')
            if query not in realQueries:
                artificial[0][query] = i
                artificial[1][i] = query
                artificial[2].append(np.array(vectors,dtype=float))
                i += 1
            else:
                real[0][query] = j
                real[1][j] = query
                real[2].append(np.array(vectors,dtype=float))
                j += 1
    return real, artificial
def generateAnnoy(real, artificial, annoyFilename):
    idx2vec = np.array(artificial[2])
    t = AnnoyIndex(EMBEDDINGDIM)
    for j in range(len(artificial[2])):
        t.add_item(j,idx2vec[j])
    print('Done Adding items to AnnoyIndex')
    t.build(TREESIZE)
    print('Done Building AnnoyIndex')
    t.save(annoyFilename)
    return t
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: makeArtificialSessions.py <realQueries> <queryVectors> <annFilename> <sessions> <outputFolder>")
        exit(-1)
    else:
        realQueries = loadQueries(sys.argv[1])
        real, artificial = loadVectors(sys.argv[2], realQueries)
        t = generateAnnoy(real, artificial, sys.argv[3])
        sessions = loadSessions(sys.argv[])