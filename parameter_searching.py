#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:04:01 2022

@author: yanyifan
"""


from my_network import *
import pickle



if __name__ == '__main__':
    hiddens = [50,100,150,200,250]
    regs = [0,0.001,0.01,0.1]
    lrs = [0.0001,0.001,0.01]
    
    best_val_acc = 0
    best_model = None
    
    result_dict = {}
    for hidden in hiddens:
        for reg in regs:
            for lr in lrs:
                net,history = training(hidden, reg, lr,verbose = False)
                val_acc = history['val acc'][-1]
                print(f'hidden={hidden},reg={reg},lr={lr}, val_acc',val_acc)
                result_dict[f'hidden={hidden},reg={reg},lr={lr}'] = val_acc
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = net
                    
    save_model(best_model, 'best_model1')
    
    
    with open('result1.txt','wb') as f:
        pickle.dump(result_dict,f)
    

    
                