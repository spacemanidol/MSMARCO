import json
def loadQueries(filename):
    queries = set()
    with open(filename,'r') as f:
        for l in f:
            queries.add(l)
    return queries    
def loadSessions(filename):
    #sessionid\tquery
    sessions = {}
    with open (filename,'r') as f:
        for l in f:
            l = l.split('\t')
            sessionID = l[0]
            query = l[1]
            if sessionID not in sessions:
                sessions[sessionID] = []
            sessions[sessionID].append(query)
    return sessions
def writeData(sessions, dirPath)
    with open(os.path.join(dirPath, 'session.tsv'),'w') as w:
        for session in sessions:
            output = ''
            for q in sessions[session]:
                output += '{}\t'.format(q)
            w.write('{}\n'.format(output[:-1]))    
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: makeArtificialSessions.py <querydir> <sessionFile> <outputFolder>")
        exit(-1)
    else:
        msmarcoQueries = loadQueries(sys.argv[1],sys.argv[2],sys.argv[3]])
        quoraQueries = loadQuora(sys.argv[4])
        nqQueries = loadNQ(sys.argv[5])
        sessions = loadSessions(sys.argv[6])
        writeData(msmarcoQueries, quoraQueries, nqQueries, sessions, sys.argv[7])