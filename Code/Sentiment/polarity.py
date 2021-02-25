# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:32:56 2017

@author: devam
"""

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
#import os

adjective_list=['JJ','JJR','JJS']
adverb_list=['RB','RBR','RBS']
verb_list=['VB','VBD','VBG','VBN','VBZ','VBP']

class PolarityCalc:

    def calculate(self, sent):
        lst=word_tokenize(sent)
        tagged=nltk.pos_tag(lst)
        pos_score=0
        neg_score=0
        word_count=0
        global_pos = 0
        global_neg = 0
        
        for w,p in tagged:
            if p in adjective_list:
                prn=list(swn.senti_synsets(w,'a'))
                if len(prn)>0:
                    
                    pos_score = prn[0].pos_score()
                    neg_score = prn[0].neg_score()
                    
                    word_count=word_count+1
                    #print(prn[0].pos_score(),prn[0].neg_score())
                
            elif p in adverb_list:
                prn=list(swn.senti_synsets(w,'r'))
                if len(prn)>0:
                    pos_score = prn[0].pos_score()
                    neg_score = prn[0].neg_score()
                    
                    word_count=word_count+1
                    #print(prn[0].pos_score(),prn[0].neg_score())
                    
            elif p in verb_list:
                prn=list(swn.senti_synsets(w,'v'))
                if len(prn)>0:
                    pos_score = prn[0].pos_score()
                    neg_score = prn[0].neg_score()
                    
                    word_count=word_count+1
                    #print(prn[0].pos_score(),prn[0].neg_score())
    #        
            if (pos_score+neg_score)!=0:
                global_pos+=pos_score
                global_neg+=neg_score
         
        
   
        if(global_pos > global_neg):
            return 1
        else:
            return -1
     
       