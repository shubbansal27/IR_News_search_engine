# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 00:23:56 2017

@author: devam
"""

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
import os
import operator
from polarity import PolarityCalc
from nltk.sentiment.vader import SentimentIntensityAnalyzer

path = 'D:\\BITS\\Sem3\\IR\\bbc'
sid = SentimentIntensityAnalyzer()
#alpha=7 #for adj
#beta=3#for adverb and verbs

abval={2:1,3:1,5:2,8:5}
for k,v in abval.items():
    alpha=k
    beta=v
    print('alpha',alpha,'beta',beta)
    for dir in os.listdir(path):
        className = dir
        print(className)
        #print('########')
        correct_prediction=0
        wrong_prediction=0
        tpos=0
        tneg=0
        fpos=0
        fneg=0
        filePath = path+'\\'+dir
        for file in os.listdir(filePath):
            fileName = file
            fPath = filePath+'\\'+file
            fp = open(fPath,'r')
            
            #converting to lower case and toeknizing
            news = fp.read().lower()
            ss = sid.polarity_scores(news)
            if ss['pos']>ss['neg']:
                vader_predicted='positive'
            else:
                vader_predicted='negative'
            #del ss['compound']
            #print(max(ss.items(), key=operator.itemgetter(1))[0])
            #print(ss)
                
            fp.close()
            #print()
        
            
            adjective_list=['JJ','JJR','JJS']
            adverb_list=['RB','RBR','RBS']
            verb_list=['VB','VBD','VBG','VBN','VBZ','VBP']
    
            sent_count=0
            filtered_sentences=[]
            pc = PolarityCalc()
            
            for sent in sent_tokenize(news):
                #print(sent)
                sent_count=sent_count+1
                lst=word_tokenize(sent)
                tagged=nltk.pos_tag(lst)
                sub_score=0
                obj_score=0
                word_count=0
                
                global_sub=0
                global_obj=0
                
                for w,p in tagged:
                    if p in adjective_list:
                        prn=list(swn.senti_synsets(w,'a'))
                        if len(prn)>0:
                            
                            sub_score = alpha*(prn[0].pos_score()+prn[0].neg_score())
                            obj_score = alpha*(1-sub_score)
                            
                            word_count=word_count+1
     
                        
                    elif p in adverb_list:
                        prn=list(swn.senti_synsets(w,'r'))
                        if len(prn)>0:
                            sub_score = beta*(prn[0].pos_score()+prn[0].neg_score())
                            obj_score = beta*(1-sub_score)
                            word_count=word_count+1
    
                            
                    elif p in verb_list:
                        prn=list(swn.senti_synsets(w,'v'))
                        if len(prn)>0:
                            sub_score = beta*(prn[0].pos_score()+prn[0].neg_score())
                            obj_score = beta*(1-sub_score)
                            word_count=word_count+1
    
                    
                    if sub_score!=0:
                        global_sub+=sub_score
                        global_obj+=obj_score
                if word_count>0:
                    global_sub=float(global_sub)/word_count
                    global_obj=float(global_obj)/word_count
    
                
                
                if global_sub>global_obj:
                    filtered_sentences.append(sent)
    
            
            i = 0
            pos=0
            neg=0
            
           
            
            for sent in filtered_sentences:
                res = pc.calculate(sent)
                if res==1:
                    pos=pos+1
                else:
                    neg=neg+1
            
            if abs(pos-neg)<=2:
                calculated_sentiment='neutral'
            elif pos>neg:
                calculated_sentiment='positive'
            else:
                calculated_sentiment='negative'
            
            if calculated_sentiment=='neutral' or vader_predicted==calculated_sentiment:
                correct_prediction=correct_prediction+1
            else:
                wrong_prediction=wrong_prediction+1
            if (vader_predicted=='positive' and calculated_sentiment=='positive') or calculated_sentiment=='neutral':
                tpos+=1
            if (vader_predicted=='positive' and calculated_sentiment=='negative'):
                fneg+=1
            if (vader_predicted=='negative' and calculated_sentiment=='negative'):
                tneg+=1
            if (vader_predicted=='negative' and calculated_sentiment=='positive'):
                fpos+=1
            
        print('tp',tpos,'fn',fneg,'tn',tneg,'fp',fpos)
        prec=float(tpos)/(tpos+fpos)
        recall=float(tpos)/(tpos+fneg)
        fmeasure=float(2*prec*recall)/(prec+recall)
        print('prec',prec*100,'recall',recall*100,'fmeasure',fmeasure*100)
        
        accuracy=float(correct_prediction)/(correct_prediction+wrong_prediction)
        print(correct_prediction,wrong_prediction,("{:.2f}".format(accuracy*100)))