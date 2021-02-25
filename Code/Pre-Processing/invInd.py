import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PrePInv import NLTKPreprocessor
import pickle


path = 'D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\data_files'
stopWords = set(stopwords.words("english"))
#fop = open('D:\\Mohit IR\\DataSets\\bbc-fulltext\\consolidated.txt','w')
lemmatizer = WordNetLemmatizer()
prep = NLTKPreprocessor()
allWords = []
invertedIndex = {}

docCount = -1

for dir in os.listdir(path):
    className = dir
#    print(className+'\n\n')
    filePath = path+'\\'+dir
    for file in os.listdir(filePath):
        fileName = file
        fPath = filePath+'\\'+file
        fp = open(fPath,'r')
        docCount+=1
        #converting to lower case and toeknizing
        fileContent = fp.read()
        fp.close()
#        print(fileContent)
        filteredContent = prep.transform(fileContent)
#        print(filteredContent)
#        print(filteredContent)
        for key in filteredContent:
            if key not in invertedIndex:
                value = []
                value.append(docCount)
            else:
                value = invertedIndex[key]
                if docCount not in value:
                    value.append(docCount)
            invertedIndex[key] = value
print(invertedIndex)
fout = open(path+'\\..\\pickles\\invertedInd.pik','wb')
pickle.dump(invertedIndex,fout)
fout.close()

        