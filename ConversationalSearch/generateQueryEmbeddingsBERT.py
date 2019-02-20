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
def process(queryPack, response):
    output = ''
    for i in range(len(queryPack)):
        output += "{}\t".format(queryPack[i])
        for j in range(len(response[i])):
            output += "{} ".format(response[i][j])
        output += "\n"
    return output
def getVectors(queries, filename):
    i = 1
    bc = BertClient()
    query = 'who founded microsoft'
    print("Testing bc\nTesting Query:{}\nVector:{}".format(query, bc.encode([query])[0]))
    with open(filename,'w') as w:
        for j in range(0,len(queries), 100):
            if i % 100 == 0:
                print('{} vectors retrieved'.format(i*100))
            queryPack = queries[j:j+100]
            response = bc.encode(queryPack)
            w.write(process(queryPack, response))
            i += 1
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: generateQueryEmbeddingsBERT.py <queryFile> <outputfile>')
        exit(-1)
    else:
        queries = loadQueries(sys.argv[1])
        print("{} queries loaded".format(len(queries)))
        getVectors(queries, sys.argv[2])
