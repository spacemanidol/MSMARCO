import sys
import json
import time
from bert_serving.client import BertClient
def loadQueries(filename):
    queries = []
    with open(filename,'r') as f:
        for l in f:
            queries.append(l.strip())
    return queries
def getVectors(queries, filename):
    i = 1
    bc = BertClient()
    query = 'who founded microsoft'
    print("Testing bc\nTesting Query:{}\nVector:{}".format(query, bc.encode([query])[0]))
    with open(filename,'w') as w:
        for query in queries:
            if i % 1000 == 0:
                print('{} vectors retrieved'.format(i))
                time.sleep(100) #my computer is small and gets too hot too quick
            try:
                w.write("{}\t{}\n".format(query, bc.encode([query])[0]))
            except:
                continue
            i += 1
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: generateQueryEmbeddingsBERT.py <queryFile> <outputfile>')
        exit(-1)
    else:
        queries = loadQueries(sys.argv[1])
        print("{} queries loaded".format(len(queries)))
        getVectors(queries, sys.argv[2])
