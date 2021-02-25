from nltk.corpus import wordnet
from PreP import NLTKPreprocessor


query = ['Firms that flout networks']
preProc = NLTKPreprocessor()
query = preProc.transform(query) 
qWords = query[0].split()
print(qWords)
dct={}
for word in qWords:
    value=[word]
    dct[word] = value
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.name() not in value: 
                value.append(l.name())
    
    dct[word]=value
#print(dct)


    
allQuery = []
def all_comb(currPos,total,newQuery):
    if(currPos>=total):
        if list(newQuery) not in allQuery:
            allQuery.append(' '.join(list(newQuery)))
        return
    for syn in dct[qWords[currPos]]:
        newQuery.append(syn)
        all_comb(currPos+1,total,newQuery)
        newQuery.pop()


def calcSim(query,exp):
    qWords = query.split()
    eWords = exp.split()
    sim = 0
    for i in range(len(qWords)):
        a = wordnet.synsets(qWords[i])
        if len(a)==0:
            sim+=1
            continue
        b = wordnet.synsets(eWords[i])
        if a[0].wup_similarity(b[0]) is not None:
            sim+= (a[0].wup_similarity(b[0]))
    return sim

all_comb(0,len(qWords),[])
print(len(allQuery))

qSim = []

for exp in allQuery:
    qSim.append(calcSim(query[0],exp))
print(qSim)

