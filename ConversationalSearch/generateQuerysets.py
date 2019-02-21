import json
import sys
import os
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
            if len(l) > 4:
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
def loadUnusedMSMARCO(filename):
    #AnswerIsInPassage\tJudgeID\tHitGroupDataInt\tHitDataInt\tHitState\tJudgmentState\tJudgmentDataInt\tJudgmentDataIntName\tJudgmentSubmitTime\tJudgmentTypeID\tTimeSpentOnJudgment\tHitGroupID\tHitID\tQueryId\tQueryText\tPassagesJson\tJudgmentID\tjudgment\tanswerTextArea\tnopassageSubmit\texplainNoAnswer\texplainCantJudge\tSL_locer\tSL_lng_from\tSL_lng_to\tJudgmentType\tGoldHitType\tGoldHitComments\t@RealTimeAuditComment\tConsensus\n'
    queries = set
    with open(filename, 'r') as f:
        for l in f:
            l = l.split('\t')
            queries.append(l[14])
    return queries
def loadSessions(filename):
    #sessionid\tquery
    sessions, queries = {}, set()
    with open (filename,'r') as f:
        for l in f:
            l = l.split('\t')
            sessionID = l[0]
            query = l[1]
            queries.add(query)
            if sessionID not in sessions:
                sessions[sessionID] = []
            sessions[sessionID].append(query)
    return sessions, queries
def writeData(msmarcoQueries, quoraQueries, nqQueries, unusedMSMARCOQueries, realQueries, sessions):
    allQueries = realQueries.union(unusedMSMARCOQueries.union(msmarcoQueries.union(quoraQueries.union(nqQueries))))
    with open('msmarcoQueries.tsv','w') as w:
        for query in msmarcoQueries:
            w.write('{}\n'.format(query))
    with open('quoraQueries.tsv','w') as w:
        for query in quoraQueries:
            w.write('{}\n'.format(query))
    with open('nqQueries.tsv','w') as w:
        for query in nqQueries:
            w.write('{}\n'.format(query))
    with open('unusedMSMARCOQueries.tsv','w') as w:
        for query in unusedMSMARCOQueries:
            w.write('{}\n'.format(query))
    with open('realQueries.tsv','w') as w:
        for query in realQueries:
            w.write('{}\n'.format(query))
    with open('allQueries.tsv','w') as w:
        for query in allQueries:
            w.write('{}\n'.format(query))
    with open('cleanedSessions.tsv','w') as w:
        for session in sessions:
            for query in sessions[session][:-1]:
                w.write('{}\t'.format(query))
            w.write('{}\n'.format(sessions[session][-1]))
if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: generateQuerysets.py <msmarco train queries> <msmarco dev queries> <msmarco eval queries> <quoraQueries> <NQFolder> <unused msmarco> <sessions>")
        exit(-1)
    else:
        msmarcoQueries = loadMSMARCO([sys.argv[1],sys.argv[2],sys.argv[3]])
        print("Done reading MSMARCO")
        quoraQueries = loadQuora(sys.argv[4])
        print("Done reading quora")
        #nqQueries = loadNQ(sys.argv[5])
        nqQueries = [] # NQ TOS do not allow us to use the dataset for this type of research
        print("Done reading NQ")
        #unusedMSMARCO = loadUnusedMSMARCO(sys.argv[6])
        unusedMSMARCO = []
        print("Done reading Unused")
        sessions, realQueries = loadSessions(sys.argv[7])
        print("Done reading Sessions")
        writeData(msmarcoQueries, quoraQueries, nqQueries, unusedMSMARCO, realQueries, sessions)
