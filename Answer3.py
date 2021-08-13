# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:11:44 2021

@author: shubh
"""
import numpy as np
import pickle
np.random.seed(1234)

class neuron():
    def __init__(self,W= np.random.random((1,2))):
        self.W=W
        
    def adder(self,I):
        v= np.dot(self.W,I)
        return v
    
    def activate(self,v):
        y= (np.sin(v))**2
        return y
    
    def learn(self,X,D,lr,iterations):
        X = X.astype(float)
        X = np.vstack((np.ones(X.shape[1]),X))
        for iter in range(iterations):
            for i in range(D.shape[0]):
                feed=X[:,i]
                d=D[i]
                # compute weighted sum of inputs
                v = self.adder(feed)
                # apply activation function
                y = self.activate(v)
                # weight update
                del_W= lr*(d-y)*2*np.sin(v)*np.cos(v)*feed
                self.W=self.W+del_W
        return self.W
    def forward(self,x):
        x = x.astype(float)
        return (self.activate(self.adder(x)))        
    
if __name__ == '__main__':
    
    # inputs
    X=np.array([[0,2,4,6,8,10]])

    # desired outputs
    D=np.array([0,1,0,1,0,1])
    
    # create a neuron with random weights
    unit= neuron()
    # learn the neuron on data
    learnt_W= unit.learn(X,D,lr=1,iterations=1000)
    
    # test the trained unit
    trained_unit=neuron(learnt_W)
    X = np.vstack((np.ones(X.shape[1]),X))
    
    for x,d in zip(X.T,D):
        print('Input:',x[1], 'Desired output:',d,'Neuron output:', trained_unit.forward(x)[0])
    
    with open('answer3.pickle', 'wb') as f:
        pickle.dump(learnt_W, f)
    
    with open('answer3.pickle', 'rb') as f:
        print(pickle.load(f))
