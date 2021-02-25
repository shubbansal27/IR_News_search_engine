import sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from PreP import NLTKPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from pathlib import Path

business = 510
entertainment = 386
politics = 417
sport = 511
tech = 401

path = 'D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\data_files'
#dataset = sklearn.datasets.load_files("D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\data_files",encoding='utf-8', decode_error='ignore')
proData = []


for dir in os.listdir(path):
    className = dir
#    print(className+'\n\n')
    filePath = path+'\\'+dir
    for file in os.listdir(filePath):
        fileName = file
        fPath = filePath+'\\'+file
        fp = open(fPath,'r')
        #converting to lower case and toeknizing
        fileContent = fp.read()
        fp.close()
        proData.append(fileContent)

picFile = Path('D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\pickles\\preProData.pik')

if picFile.exists():
    fin = open('D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\pickles\\preProData.pik','rb')
    proData = pickle.load(fin)
    fin.close()
else:
    prePro = NLTKPreprocessor()
    proData = prePro.transform(proData)
    fout = open('D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\pickles\\preProData.pik','wb')
    pickle.dump(proData,fout)
    fout.close()
    
    
query = ['Firms that flout networks']
preProc = NLTKPreprocessor()
query = preProc.transform(query) 
qWords = query[0].split()
fin = open('D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\pickles\\invertedInd.pik','rb')
invertedIndex = pickle.load(fin)
fin.close()
docList = []
for word in qWords:
    if word in invertedIndex:
        for doc in invertedIndex[word]:
            if doc not in docList:
                docList.append(doc)

proDataFiltered = []
for index in docList:
    proDataFiltered.append(proData[index]) 
    
    
    
    
    
#print(dataset.target)
#print(proData)
count_vect = CountVectorizer()
cVector = count_vect.fit_transform(proDataFiltered)
#print(cVector)
tfidf_transformer = TfidfTransformer()
tVector = tfidf_transformer.fit_transform(cVector)


cQuery = count_vect.transform(query)
tQuery = tfidf_transformer.transform(cQuery)
print(tQuery)

sim = cosine_similarity(tQuery, tVector)
print(sim)

simScore = []
docCount = 0
for score in list(sim[0]):
    scoreList = []
    docCount+=1
    scoreList.append(score)
    scoreList.append(docList[docCount-1])
    simScore.append(scoreList)
simScore.sort(key=lambda x: x[0],reverse=True)
#print(simScore)
#print(simScore)

def readFile(className,docNo):
    count = 0
    for file in os.listdir(path+'\\'+className):
        count+=1
        if count==docNo:
            fin = open(path+'\\'+className+'\\'+file,'r')
            content = fin.read()
            fin.close()
            return (className,file,content)
    return None


resultList = []
print(simScore[0:10])

for item in simScore[0:10]:
    docNo = item[1]+1
    if docNo>business:
        docNo=docNo-business
    else:
        resultList.append(readFile('business',docNo))
        continue
    if docNo>entertainment:
        docNo=docNo-entertainment
    else:
        resultList.append(readFile('entertainment',docNo))
        continue
    if docNo>politics:
        docNo=docNo-politics
    else:
        resultList.append(readFile('politics',docNo))
        continue
    if docNo>sport:
        docNo=docNo-sport
    else:
        resultList.append(readFile('sport',docNo))
        continue
    resultList.append(readFile('tech',docNo))
#print(resultList)
    
#print(type(a))
#a.sort(reverse=True)
#print(a[0:9])





#print(tVector)
#fout = open(path+'\\..\\pickles\\tfIdf.pik','wb')
#pickle.dump(tVector,fout)
#fout.close()