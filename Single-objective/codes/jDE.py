# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:42:16 2018

Parameters adaption Diffierential Evolution: jDE

Input:   population matrix (NP * D)
Output:  population matrix (NP * D)

@author: EuroBrother
"""
import numpy as np

#def benchmark_func(vec):
#    dim = len(vec)
#    f = 100 * np.sum((vec[0:dim-1]**2 - vec[1:dim])**2) + np.sum((vec[0:dim-1] - 1)**2)
#    
#    return f

#####################
def optimizer(f,ub,lb,Fv,CRv,max_fes,XTemp,vec,index):
#max_fes = 3*10**5
#D = 30

#    lb = -30
#    ub = 30
    #NP = 30
    pop = XTemp
    NP,D = np.shape(XTemp)
    popold = np.zeros((NP,D))
    t1 = 0.1
    t2 = 0.1
    
    F = Fv * np.ones((NP,1))    # Fv = 0.5
    CR = CRv * np.ones((NP,1))  # CRv = 0.9
    
    
    val = np.zeros(NP)
    for i in range(NP):
        for j in range(D):
            if pop[i,j] < lb:
                pop[i,j] = 2 * lb - pop[i,j]
            if pop[i,j] > ub:
                pop[i,j] = 2 * ub - pop[i,j]
        vec[index] = pop[i,:]
        val[i] = f.do_evaluate(vec)
    n_fes = 0
    
    pm1 = np.zeros((NP,D))
    pm2 = np.zeros((NP,D))
    pm3 = np.zeros((NP,D))
    ui = np.zeros((NP,D+2))
    mui = np.zeros((NP,D))
    mpo = np.zeros((NP,D))
    rot = np.arange(0,NP,1)
    rt = np.zeros(NP)
    a1 = np.zeros(NP)
    a2 = np.zeros(NP)
    a3 = np.zeros(NP)
    
    pop = np.hstack([pop,F])
    pop = np.hstack([pop,CR])
    g_n = 0
    res = np.ones(max_fes)
    while (g_n < max_fes):
#        print(g_n,"and",min(val))
#        indmin = np.argmin(val)
#        bestsol = pop[indmin,0:D]
#        r2 = np.linalg.norm(bestsol - np.array(f.min_loc)[index]) / 100
        res[g_n] = np.min(val)
        g_n += 1
        popold = pop[:,0:D]
        
        ind = np.random.permutation(2) + 1
        
        a1 = np.random.permutation(NP)
        rt = np.fmod(rot+ind[0],NP)
        a2 = a1[rt]
        rt = np.fmod(rot+ind[1],NP)
        a3 = a2[rt]
        
        pm1 = popold[a1,:]
        pm2 = popold[a2,:]
        pm3 = popold[a3,:]
        
        for i in range(NP):
            if np.random.rand()<t1:
                ui[i,D] = 0.1 + np.random.rand()*0.9
            else:
                ui[i,D] = pop[i,D]
            if np.random.rand()<t2:
                ui[i,D+1] = np.random.rand()
            else:
                ui[i,D+1] = pop[i,D+1]
                
        F = ui[:,D]
        CR = ui[:,D+1]
        mui = np.random.rand(NP,D) < np.tile(CR,(D,1)).T
        mui = mui + 0
        
        dd = np.floor(D*np.random.rand(NP))
        for kk in range(NP):
            mui[kk,int(dd[kk])] = 1
        mpo = mui < 0.5
        
        ui[:,0:D] = pm3 + np.tile(F,(D,1)).T * (pm1 - pm2)
        ui[:,0:D] = popold * mpo + ui[:,0:D] * mui
        
        for i in range(NP):
            for j in range(D):
                if ui[i,j] < lb:
                    ui[i,j] = 2 * lb - ui[i,j]
                if ui[i,j] > ub:
                    ui[i,j] = 2 * ub - ui[i,j]
        
        for i in range(NP):
            vec[index] = ui[i,0:D]
            tempval = f.do_evaluate(vec)
            n_fes +=1
            if (tempval <= val[i]):
                pop[i,:] = ui[i,:]
                val[i] = tempval
#        if (np.min(val) - f.fmin) < 1e-10:
#            break
    
    indmin = np.argmin(val)
    bestsol = pop[indmin,0:D]            
    return bestsol, res

def oop(f,ub,lb,Fv,CRv,max_fes,XTemp):
#max_fes = 3*10**5
#D = 30

#    lb = -30
#    ub = 30
    #NP = 30
    pop = XTemp
    NP,D = np.shape(XTemp)
    popold = np.zeros((NP,D))
    t1 = 0.1
    t2 = 0.1
    
    F = Fv * np.ones((NP,1))    # Fv = 0.5
    CR = CRv * np.ones((NP,1))  # CRv = 0.9
    
    
    val = np.zeros(NP)
    for i in range(NP):
        for j in range(D):
            if pop[i,j] < lb:
                pop[i,j] = 2 * lb - pop[i,j]
            if pop[i,j] > ub:
                pop[i,j] = 2 * ub - pop[i,j]
        val[i] = f.do_evaluate(pop[i,:])
    n_fes = 0
    
    pm1 = np.zeros((NP,D))
    pm2 = np.zeros((NP,D))
    pm3 = np.zeros((NP,D))
    ui = np.zeros((NP,D+2))
    mui = np.zeros((NP,D))
    mpo = np.zeros((NP,D))
    rot = np.arange(0,NP,1)
    rt = np.zeros(NP)
    a1 = np.zeros(NP)
    a2 = np.zeros(NP)
    a3 = np.zeros(NP)
    
    pop = np.hstack([pop,F])
    pop = np.hstack([pop,CR])
    g_n = 0
    res = np.ones(max_fes)
    while (g_n < max_fes):
#        print(g_n,"and",min(val))
#        indmin = np.argmin(val)
#        bestsol = pop[indmin,0:D]
#        r2 = np.linalg.norm(bestsol - np.array(f.min_loc)) / 1000
        res[g_n] = np.min(val)
        g_n += 1
        popold = pop[:,0:D]
        
        ind = np.random.permutation(2) + 1
        
        a1 = np.random.permutation(NP)
        rt = np.fmod(rot+ind[0],NP)
        a2 = a1[rt]
        rt = np.fmod(rot+ind[1],NP)
        a3 = a2[rt]
        
        pm1 = popold[a1,:]
        pm2 = popold[a2,:]
        pm3 = popold[a3,:]
        
        for i in range(NP):
            if np.random.rand()<t1:
                ui[i,D] = 0.1 + np.random.rand()*0.9
            else:
                ui[i,D] = pop[i,D]
            if np.random.rand()<t2:
                ui[i,D+1] = np.random.rand()
            else:
                ui[i,D+1] = pop[i,D+1]
                
        F = ui[:,D]
        CR = ui[:,D+1]
        mui = np.random.rand(NP,D) < np.tile(CR,(D,1)).T
        mui = mui + 0
        
        dd = np.floor(D*np.random.rand(NP))
        for kk in range(NP):
            mui[kk,int(dd[kk])] = 1
        mpo = mui < 0.5
        
        ui[:,0:D] = pm3 + np.tile(F,(D,1)).T * (pm1 - pm2)
        ui[:,0:D] = popold * mpo + ui[:,0:D] * mui
        
        for i in range(NP):
            for j in range(D):
                if ui[i,j] < lb:
                    ui[i,j] = 2 * lb - ui[i,j]
                if ui[i,j] > ub:
                    ui[i,j] = 2 * ub - ui[i,j]
        
        for i in range(NP):
            tempval = f.do_evaluate(ui[i,0:D])
            n_fes +=1
            if (tempval <= val[i]):
                pop[i,:] = ui[i,:]
                val[i] = tempval
#        if (np.min(val) - f.fmin) < 1e-10:
#            break
    
    indmin = np.argmin(val)
    bestsol = pop[indmin,0:D]            
    return bestsol, res

def RMS(f,dim,ub,lb,num,max_fes):
    val = np.ones(num)
    X = np.random.uniform(lb,ub,(num,dim))
    for j in range(num):
        val[j] = f.do_evaluate(X[j,:])
    best = np.min(val)
    res = np.ones(max_fes)
    for i in range(max_fes):
        X = np.random.uniform(lb,ub,(num,dim))
        for j in range(num):
            val[j] = f.do_evaluate(X[j,:])
        fb = np.min(val)
        if fb < best:
            res[i] = fb
        else:
            res[i] = best
    return res
        
           
