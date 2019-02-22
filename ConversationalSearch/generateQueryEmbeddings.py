import sys
import json
import requests
def loadQueries(filename):
    queries = []
    with open(filename,'r') as f:
        for l in f:
            queries.append(l.strip())
    return queries

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
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage:generateQueryEmbeddings.py <url> <queryFile> <outputfile>')
        exit(-1)
    else:
        queries = loadQueries(sys.argv[2])
        getVectors(queries, sys.argv[1], sys.argv[3])
