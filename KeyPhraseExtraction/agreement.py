
import numpy as np
import random
import re
ANNOTATIONFILE = 'OpenKPAnnotations.tsv'

def getScoreUnigram(candidate, gold):
    #Unigram Levenshtein distance
    #First we produce all possible pairs and greedily select 
    scoring, bestMatch = {}, {}
    maxScore = 0
    maxLabel = ''
    #Generate all possible combinations
    for goldLabel in gold:
        goldKey = str(goldLabel)
        scoring[goldKey] = {}
        for candidateLabel in candidate:
            candidateKey = str(candidateLabel)
            scoring[goldKey][candidateKey] = (len(goldLabel) - len(goldLabel-candidateLabel))/len(goldLabel)
    #Greedily select best combination and then remove all related combinations.
    while len(scoring) > 0:
        maxScore = 0
        maxLabel = ''
        for goldLabel in scoring:
            goldKey = str(goldLabel)
            for candidateLabel in scoring[goldKey]:
                candidateKey = str(candidateLabel)
                score = scoring[goldKey][candidateKey]
                if score >= maxScore:
                    maxScore = score
                    maxLabel = (goldKey, candidateKey)
        bestMatch[maxLabel] = scoring[maxLabel[0]][maxLabel[1]]
        scoring.pop(maxLabel[0])#remove all pairs that could
    return sum(bestMatch.values())/len(gold)      

def getScoreEM(candidate, gold):
    #Unigram Levenshtein distance
    #First we produce all possible pairs and greedily select 
    scoring, bestMatch = {}, {}
    maxScore = 0
    maxLabel = ''
    #Generate all possible combinations
    for goldLabel in gold:
        goldKey = str(goldLabel)
        scoring[goldKey] = {}
        for candidateLabel in candidate:
            candidateKey = str(candidateLabel)
            if goldLabel == candidateLabel:
                scoring[goldKey][candidateKey] = 1
            else:
                scoring[goldKey][candidateKey] = 0
    #Greedily select best combination and then remove all related combinations.
    while len(scoring) > 0:
        maxScore = -1
        maxLabel = ''
        for goldLabel in scoring:
            goldKey = str(goldLabel)
            for candidateLabel in scoring[goldKey]:
                candidateKey = str(candidateLabel)
                score = scoring[goldKey][candidateKey]
                if score >= maxScore:
                    maxScore = score
                    maxLabel = (goldKey, candidateKey)
        bestMatch[maxLabel] = scoring[maxLabel[0]][maxLabel[1]]
        scoring.pop(maxLabel[0])#remove all pairs that could
    return sum(bestMatch.values())/len(gold) 

def calculateAgreement(judgements):
    scoresUnigram = []
    scoresEM = []
    for url in judgements:
        for i in range(len(judgements[url])):
            currentRunsUnigram, currentRunsEM = [], []
            for j in range(len(judgements[url])):
                if j != i:
                    currentRunsUnigram.append(getScoreUnigram(judgements[url][i],judgements[url][j]))
                    currentRunsEM.append(getScoreEM(judgements[url][i],judgements[url][j]))
            if len(currentRunsUnigram) > 0:
                scoresUnigram.append(np.sum(currentRunsUnigram)/(len(judgements[url])-1))
                scoresEM.append(np.sum(currentRunsEM)/(len(judgements[url])-1))
    print('Exact Match max:{} min:{} mean:{}'.format(np.max(scoresEM), np.min(scoresEM),np.mean(scoresEM)))
    print('Unigram max:{} min:{} mean:{}'.format(np.max(scoresUnigram), np.min(scoresUnigram),np.mean(scoresUnigram)))

def loadURLs():
    urls = {}
    with open(ANNOTATIONFILE,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            url = l[14]
            if url not in urls:
                urls[url] = 0
            urls[url] += 1
    return urls
def loadJudgements(urls):
    kp1, kp2, kp3 = {}, {}, {}
    with open(ANNOTATIONFILE,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            url = l[14]
            if urls[url] > 1 and len(l) > 16:
                if url not in kp1 and 'error' not in l[16].strip().lower():
                    kp1[url] = []
                    kp2[url] = []
                    kp3[url] = []
                if 'error' not in l[16].strip().lower(): 
                    currentkp1,currentkp2, currentkp3 = [] , [], []
                    for phrase in l[16]: #top keyphrase
                        currentkp1.append(set(re.sub(' +', ' ',phrase.strip()).split(' ')))
                    for phrase in l[16:17]: #top 2 keyphrases
                        currentkp2.append(set(re.sub(' +', ' ',phrase.strip()).split(' ')))
                    for phrase in l[16:]: #top 3 keyphrases
                        currentkp3.append(set(re.sub(' +', ' ',phrase.strip()).split(' ')))
                    kp1[url].append(currentkp1)
                    kp2[url].append(currentkp2)
                    kp3[url].append(currentkp3)
    return kp1,kp2, kp3
if __name__ == "__main__":
    urls = loadURLs()
    kp1, kp2, kp3 = loadJudgements(urls)
    print("Pairwise Agreement @ Top KP")
    calculateAgreement(kp1)
    print("Pairwise Agreement @ Top 2 KP")
    calculateAgreement(kp2)
    print("Pairwise Agreement @ Top 3 KP")
    calculateAgreement(kp3)
    #print(kp3) #5 sets of KPs for each url. Used for our qualitative evaluation