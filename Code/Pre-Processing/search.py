from PrePInv import NLTKPreprocessor
import pickle


prepro = NLTKPreprocessor()
query = 'ddos attack is an the popular lauda'
qWords = prepro.transform(query)
fin = open('D:\\Mohit IR\\DataSets\\BBC-Dataset-News-Classification-master\\BBC-Dataset-News-Classification-master\\dataset\\pickles\\invertedInd.pik','rb')
invertedIndex = pickle.load(fin)
fin.close()
docList = set()
for word in qWords:
    if word in invertedIndex:
        for doc in invertedIndex[word]:
            docList.add(doc)
print(docList)