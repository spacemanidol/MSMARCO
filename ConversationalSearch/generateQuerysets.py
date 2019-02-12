import json
def loadNQ(dirPath):
    files = os.listdir(dirPath)
    queries = set()
    for file in files:
        with open(os.path.join(dirPath, file),'r') as f:
            for l in f:
                j = json.loads(l)
                query_text = j['question_text']
                queries.add(query_text)
    return queries
def loadQuora(filename):
    #id      qid1    qid2    question1       question2       is_duplicate
    #0       1       2       What is the step by step guide to invest in share market in india?      What is the step by step guide to invest in share market?       0
    queries = set()
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            queries.add(l[3])
            queries.add(l[4])
    return queries
def loadMSMARCO(files):
    #1048578 cost of endless pools/swim spa
    queries = set()
    for file in files:
        with open(file,'r') as f:
            for l in f:
                queries.add(l.strip().split('\t')[1])
    return queries       
def writeData(msmarcoQueries, quoraQueries, nqQueries, dirPath)
    allQueries = msmarcoQueries + quoraQueries + nqQueries
    print("There are {} unique MSMARCO Queries, {} unique Quora Duplicate Queries, {} unique NQ Queries for a total of {} unique queries.".format(len(msmarcoQueries), len(quoraQueries), len(nqQueries), len(allQueries)))
    with open(os.path.join(dirPath, 'msmarcoQueries.tsv'),'w') as w:
        for query in msmarcoQueries:
            w.write('{}\n'.format(query))
    with open(os.path.join(dirPath, 'quoraQueries.tsv'),'w') as w:
        for query in quoraQueries:
            w.write('{}\n'.format(query))
    with open(os.path.join(dirPath, 'nqQueries.tsv'),'w') as w:
        for query in nqQueries:
            w.write('{}\n'.format(query))
    with open(os.path.join(dirPath, 'allQueries.tsv'),'w') as w:
        for query in allQueries:
            w.write('{}\n'.format(query))
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: generateQuerysets.py <msmarco train queries> <msmarco dev queries> <msmarco eval queries> <quoraQueries> <NQFolder> <outputFolder>")
        exit(-1)
    else:
        msmarcoQueries = loadMSMARCO([sys.argv[1],sys.argv[2],sys.argv[3]])
        quoraQueries = loadQuora(sys.argv[4])
        nqQueries = loadNQ(sys.argv[5])
        writeData(msmarcoQueries, quoraQueries, nqQueries, sys.argv[6])