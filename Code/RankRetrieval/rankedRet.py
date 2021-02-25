import sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from modules.RankRetrieval.PreP import NLTKPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from pathlib import Path
from nltk.corpus import wordnet


'''variables'''

business = 510
entertainment = 386
politics = 417
sport = 511
tech = 401
path = 'Dataset\\bbc'


'''utility functions'''

def readFile(className,docNo):
    count = 0
    for file in os.listdir(path+'\\'+className):
        count+=1
        if count==docNo:
            fin = open(path+'\\'+className+'\\'+file,'r')
            content = fin.read()
            fin.close()
            #return (className,file,content)
            return (className,file)
    return None

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


def search_main(keyword):
  
   proData = []
   query = [keyword]
   allSim =[]
  
   '''reading dataset'''
    
   for dir in os.listdir(path):
        #className = dir
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
    
   picFile = Path('pickles\\bbc\\preProData.pik')
    
   if picFile.exists():
        fin = open('pickles\\bbc\\preProData.pik','rb')
        proData = pickle.load(fin)
        fin.close()
   else:
        prePro = NLTKPreprocessor()
        proData = prePro.transform(proData)
        fout = open('pickles\\bbc\\preProData.pik','wb')
        pickle.dump(proData,fout)
        fout.close()
        
    
   fin = open('pickles\\bbc\\invertedInd.pik','rb')
   invertedIndex = pickle.load(fin)
   fin.close()
    
    
   '''query expansion'''
    
   preProc = NLTKPreprocessor()
   query = preProc.transform(query) 
   qWords = query[0].split()
   dct={}
   for word in qWords:
       value=[word]
       dct[word] = value
       for syn in wordnet.synsets(word):
           for l in syn.lemmas():
               if l.name() not in value: 
                   value.append(l.name())
        
       dct[word]=value
    
   allQuery = []


   def all_comb(currPos,total,newQuery):
#    print('In rec')
    if(currPos>=total):
        if list(newQuery) not in allQuery:
            allQuery.append(' '.join(list(newQuery)))
        return
    for syn in dct[qWords[currPos]]:
        newQuery.append(syn)
        all_comb(currPos+1,total,newQuery)
        newQuery.pop()
        
        
   all_comb(0,len(qWords),[])
    
   qSim = []
    
   for exp in allQuery:
       qSim.append(calcSim(query[0],exp))
    
   docQInv = {}
   i = -1
   for query in allQuery:
       i+=1
       qWords = query.split()
       docList = []
       for word in qWords:
           if word in invertedIndex:
               for doc in invertedIndex[word]:
                   if doc not in docList:
                       docList.append(doc)
       proDataFiltered = []
       for index in docList:
           proDataFiltered.append(proData[index]) 
       if len(proDataFiltered)==0:
           continue
    
       count_vect = CountVectorizer()
       cVector = count_vect.fit_transform(proDataFiltered)
       tfidf_transformer = TfidfTransformer()
       tVector = tfidf_transformer.fit_transform(cVector)
       cQuery = count_vect.transform([query])
       tQuery = tfidf_transformer.transform(cQuery)
    
       sim = cosine_similarity(tQuery, tVector)
        
       simScore = []
       docCount = 0
       for score in list(sim[0]):
           scoreList = []
           docCount+=1
           scoreList.append(score)
           scoreList.append(docList[docCount-1])
           simScore.append(scoreList)
           simScore.sort(key=lambda x: x[0],reverse=True)
       allSim.append(simScore[0:9])   
       resultList = []
    
       for item in simScore[0:9]:
           docNo = item[1]+1
           if docNo not in docQInv:
               docQInv[docNo] = [[i,item[0]]]
           else:
               docQInv[docNo].append([i,item[0]])
    
   docScore = []
   for doc in docQInv.keys():
       tScore = 0
       for elem in docQInv[doc]:
           qNo = elem[0]
           dScore = elem[1]
           qScore = qSim[qNo]
           tScore+= dScore*qScore
       docScore.append([tScore,doc])
    
   docScore.sort(key=lambda x: x[0],reverse=True)
   #print(docScore)
   
   for item in docScore[0:9]:
    docNo = item[1]
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
    
   return resultList 


