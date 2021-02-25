
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
#fop = open('D:\\BITS\\Sem3\\IR\\sport.txt','w')

glbl = {}
#current={'win':'positive', 'lose':'negative', 'good':'positive', 'power':'positive','strong':'positive', 'injury':'negative', 'record':'positive', 'drug':'negative', 'cheat':'negative', 'ban':'negative'}

current={'good':'positive'}

temp={}
flag = True
prev=0
stopWords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


while flag:
    for word,v in current.items():

        for syn in wordnet.synsets(word):
                print(syn)
                for l in syn.lemmas():
                    curr=lemmatizer.lemmatize(l.name())
                    temp[curr]=v              
                    #print(curr)
                    if l.antonyms():
                        #print(l.antonyms())
                        curra=lemmatizer.lemmatize(l.antonyms()[0].name())
                        print(curra)
                        if(v=='positive'):
                            temp[curra]='negative' 
                        else:
                            temp[curra]='positive' 
                                              
                    
    glbl.update(current)
    current.clear()
    current.update(temp)
    if(len(glbl)!=prev):
#        print('#############')
#        print(len(glbl))
        prev=len(glbl)
    else:
        break
    
#print(glbl)

path = 'D:\\BITS\\Sem3\\IR\\bbc\\sport'

#news="Dibaba breaks 5,000m world record Ethiopia's Tirunesh Dibaba set a new world record in winning the women's 5,000m at the Boston Indoor Games. Dibaba won in 14 minutes 32.93 seconds to erase the previous world indoor mark of 14:39.29 set by another Ethiopian, Berhane Adera, in Stuttgart last year. But compatriot Kenenisa Bekele's record hopes were dashed when he miscounted his laps in the men's 3,000m and staged his sprint finish a lap too soon. Ireland's Alistair Cragg won in 7:39.89 as Bekele battled to second in 7:41.42.  I didn't want to sit back and get out-kicked,  said Cragg.  So I kept on the pace. The plan was to go with 500m to go no matter what, but when Bekele made the mistake that was it. The race was mine.  Sweden's Carolina Kluft, the Olympic heptathlon champion, and Slovenia's Jolanda Ceplak had winning performances, too. Kluft took the long jump at 6.63m, while Ceplak easily won the women's 800m in 2:01.52."


totalpos=0
totalneg=0

for file in os.listdir(path):
    fileName = file
    fPath = path+'\\'+file
    fp = open(fPath,'r')

    news=fp.read().lower()
    fp.close()
    tokenizedContent = word_tokenize(news)
    filteredContent = []
    for word in tokenizedContent:
        if word not in stopWords:
            filteredContent.append(word)
            
            
    lemmContent = []
    for word in filteredContent:
        lemmContent.append(lemmatizer.lemmatize(word))
    
    #print(lemmContent)
     
    positive=0
    negative=0
    absent=0
    for keyword in lemmContent:
        if keyword in glbl.keys():
            if glbl[keyword]=='positive':
                positive=positive+1
            else:
                negative=negative+1
        else:
            absent=absent+1
    
    #positive=float(positive)/(positive+negative+absent)
    #negative=float(negative)/(positive+negative+absent)
    
    if positive>negative:
        cls='positive'
        totalpos=totalpos+1
        
    else:
        cls='negative'
        totalneg=totalneg+1
    if cls=='negative':
        print('file',fileName,'--> ',positive,negative,'  class=',cls)
        
print(totalpos,totalneg)


        












#syns=wordnet.synsets("field")

#print(syns)
#print(syns[0].lemmas()[0].name())
#print(syns[0].definition())


