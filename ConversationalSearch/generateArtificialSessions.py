import json
import math
import numpy as np
import sys
from annoy import AnnoyIndex

TREESIZE = 1000

def loadQueries(filename):
    queries = set()
    with open(filename,'r') as f:
        for l in f:
            queries.add(l.strip())
    return queries

def loadSessions(filename):
    sessions = []
    with open(filename,'r') as f:
        for l in f:
            sessions.append(l.strip().split('\t'))
    return sessions

def loadVectors(filename, realQueries):
    i = 0
    j = 0
    artificial = [{},{},[]] #Query2Idx, idX2Query, id2Vector
    real = [{},{},[]] #Query2Idx, idX2Query, id2Vector
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

def generateAnnoy(real, artificial, annoyFilename, dimensions):
    idx2vec = np.array(artificial[2])
    t = AnnoyIndex(dimensions)
    for j in range(len(artificial[2])):
        t.add_item(j,idx2vec[j])
    print('Done Adding items to AnnoyIndex')
    t.build(TREESIZE)
    print('Done Building AnnoyIndex')
    t.save(annoyFilename)
    return t

def generateArtificialSessions(queryVectors, sessions, annoyEmbedding, filename):
    with open(filename,'w') as w:
        for session in sessions:
            queriesUsed = set()
            output = ''
            for query in sessions[session]:
                vectorIndex = queryVectors[0][query]
                v = queryVectors[2][vectorIndex]
                artificialQueries = annoyEmbedding.get_nns_by_vector(v, 10, search_k=-1, include_distances=False)
                for artificialQuery in artificialQueries:
                    if artificialQuery not in queriesUsed: #ensure session isnt just repeating queries
                        queriesUsed.add(artificialQuery)
                        output += '{}\t'.format(artificialQuery)
                        break
            w.write("{}\n".format(output[:-1]))
    
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: makeArtificialSessions.py <realQueries> <queryVectors> <BERTVectors> <annFilename> <sessions>")
        exit(-1)
    else:
        print("Loading Queries and Sessions")
        realQueries = loadQueries(sys.argv[1])
        print("Loading Sessions")
        sessions = loadSessions(sys.argv[5])
        #Run regular embeddings
        print("Loading Query Vectors")
        real, artificial = loadVectors(sys.argv[2], realQueries)
        print("Building Annnoy Query Embeddings")
        annoyEmbedding = generateAnnoy(real, artificial, sys.argv[4], 100)
        print("Generating Sessions Query Embeddings")
        generateArtificialSessions(artificial, sessions, annoyEmbedding, 'sessionsEmbedding.tsv')
        #Run on BERT embeddings
        print("Loading BERT Vectors")
        real, artificial = loadVectors(sys.argv[3], realQueries)
        print("Building Annnoy Query Embeddings")
        annoyEmbedding = generateAnnoy(real, artificial, 'BERT' + sys.argv[4], 1024)
        print("Generating Sessions Query Embeddings")
        generateArtificialSessions(artificial, sessions, annoyEmbedding, 'sessionsEmbeddingBERT.tsv')
