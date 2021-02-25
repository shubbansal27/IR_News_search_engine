# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:47:23 2017

@author: SWD
"""
import pickle
favorite_color = { "lion": "yellow", "kitty": "red" } 
pickle.dump( favorite_color, open( "save.p", "wb" ) )