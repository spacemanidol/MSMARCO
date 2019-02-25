import json
import math
import numpy as np
import sys
import requests

def generateOuput(responses):
    output = ''
    for response in responses:
        output += '{}\t'.format(response['query'])
        for v in response['vector'][:-1]:
            output += '{} '.format(v)
        output += '{}\n'.format(response['vector'][-1])
    return output

def getVectors(queries, key, filename):
    chunks = []
    count = 1
    for i in range(0, len(queries), 10): 
        chunks.append(queries[i:i+10])
    with open(filename,'w') as w:
        for i, chunk in enumerate(chunks):
            if i % 100 == 0:
                print('{} vectors retrieved'.format(i*10))
            i += 1
            try:
                w.write(generateOuput(requests.get(url=key + str(chunk) + "}").json()))
            except:
                try:
                    w.write(generateOuput(requests.get(url=key + str(chunk) + "}").json()))
                except:
                    try:
                        w.write(generateOuput(requests.get(url=key + str(chunk) + "}").json()))
                    except:
                        #Its dirty but sometimes the API fails and this is the easiest fix
                        continue


def loadQueries(filename):
    queries = set()
    with open(filename,'r') as f:
        for l in f:
            queries.add(l.strip())
    return queries
def loadVectors(filename, realQueries, artificialQueries):
    i = 0
    j = 0
    artificial = [{},{},[]] #Query2Idx, idX2Query, id2Vector
    real = [{},{},[]] #Query2Idx, idX2Query, 
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            query = l[0]
            vectors = l[1].split(' ')
            if query in artificialQueries:
                artificial[0][query] = i
                artificial[1][i] = query
                artificial[2].append(np.array(vectors,dtype=float))
                i += 1
            if query in realQueries:
                real[0][query] = j
                real[1][j] = query
                real[2].append(np.array(vectors,dtype=float))
                j += 1
    return real, artificial

realQueries = loadQueries(sys.argv[1])
artificialQueries = loadQueries(sys.argv[2])
real, artificial = loadVectors(sys.argv[3], realQueries, artificialQueries)
a = set()
for q in realQueries:
    if q not in real[0]:
        a.add(q)
    

for q in artificialQueries:
    if q not in artificial[0]:
        a.add(q)


a = list(a)
getVectors(a, url, 'missing')
