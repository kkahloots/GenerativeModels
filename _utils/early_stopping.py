#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:52:46 2018

@author: pablosanchez
"""
import _utils.constants as const

class EarlyStopping(object):
    def __init__(self, name='', patience=15, min_delta = const.tol):
        self.name = name
        self.patience = patience
        self.min_delta = min_delta
        self.patience_cnt = 0
        self.prev_loss_val = 200000000
        
    
    def stop(self, loss_val):
        if(self.prev_loss_val - loss_val> self.min_delta):
            self.patience_cnt = 0
            self.prev_loss_val = loss_val
            
        else:
            self.patience_cnt += 1
            print(self.name + ' Patience count: ', self.patience_cnt)
            
        if(self.patience_cnt > self.patience):
            return True
        else:
            return False
        
    