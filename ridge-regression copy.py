#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:41:08 2018

@author: marziehmehdizadeh
"""
import numpy as np
from numpy import pi, exp, log
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import math

#Question1
class RidgeRegression(object):
    def initialize (self, lmbd, eta):
        self.lmbd=lmbd
        self.eta=eta
        

    def calc_w(self, X,Y):
        self.X = np.transpose(np.asmatrix(X))
        self.Y= np.transpose(np.asmatrix(Y))
        
    
        #builing optimom solution for w, here .eye gives an identitiy matrix of dim X.
        #self.XT= np.transpose(self.X)
        c = np.matmul(np.transpose(self.X),self.X) + self.lmbd*np.eye(self.X.shape[1])
        d=np.linalg.inv(c)
        e=np.matmul(np.transpose(self.X),self.Y)
        self.w_calc = np.array(np.matmul(d,e))

# gradient decent
    '''def gradient_decent(self,eta,epsilon):
        cond = True
        while cond == True:
            w_t1 = self.w + eta * (np.dot(self.XT,(self.Y - np.dot(self.X,self.w))) - self.lmbd * self.w)
            sum_error = 0.0
            for i in range(len(self.w[0])):
                sum_error += float(abs(self.w[0][i] - w_t1[0][i]))
                print(sum_error)
                if (sum_error <= epsilon):
                    cond = False
                self.w = w_t1
    def report(self):
        return self'''
    
    def gradientDesc(self,X,Y, eta, epsilon,n_iter, w_init):
        cond=True
        i = 0
        while cond==True and i < n_iter:
            w_t1 = w_init - 2*eta * (np.dot(np.transpose(self.X),np.dot(self.X,w_init)) - 
                                     np.dot(np.transpose(self.X),self.Y) + self.lmbd * w_init)
            #print(w_t1)
            # print(w_init)
            # print(self.X)
            # print(self.Y)
            sum_error = float(abs(w_init - w_t1))
            
            #print(sum_error)
            if (sum_error <= epsilon):
                cond = False
            w_init=w_t1
            i+= 1
            #print(i)
            #print('^^^^')
        self.w = w_t1
        
    def predict(self,b,X):
        self.Y_hat=self.X.dot(self.w)+b
        #print(self.Y_hat)
    def lossfunc(self, Y):
        self.Y=np.array(self.Y)
        self.Y_hat=np.array(self.Y_hat)
        #self.cost = math.sqrt(sum((self.Y-self.Y_hat)**2))
        self.cost=np.square(self.Y - self.Y_hat).mean()
        return(self.cost)
    def result(self):
        return self
    
    def plot(self):
        t = np.linspace(-10,10,100) 
        fig, ax = plt.subplots()
        
        # Using set_dashes() to modify dashing of an existing line
        line1, = ax.plot(self.X, self.Y, "o", label='Obervations')
        

        # Using plot(..., dashes=...) to set the dashing when creating a line
        line2, = ax.plot(self.X, self.Y_hat, 'd', label='Predictions')
        line3,=ax.plot(t, np.sin(t)+0.3*t-1.0, label='Curve' )
        plt.title( 'Lambda = %s' % (lmbd))
        ax.legend()
        plt.show()
        
    
        

#%%%
#Question 2
X_train=np.random.uniform(low=-5, high=5, size=15) 
Y_train=np.sin(X_train)+0.3*X_train-1
D_n=[X_train,Y_train]

#plt.plot(X_train,Y_train, "o")
#plt.show()
#%%
b = -1 # Y(X=0)
eta = 0.001
epsilon = 1.0e-6
n_iter= 1000
w_init = 0.5
RR = RidgeRegression()  
    

#%%
#Question3
#b = -1 # Y(X=0)
lmbd = 0.0

RR.initialize(lmbd,eta)
RR.calc_w(X_train,Y_train)
RR.gradientDesc(X_train, Y_train,eta, epsilon,n_iter, w_init)
RR.predict(b,X_train)
result = RR.result()
#print(result.w_calc)
#print (result.w)
RR.plot()




#Question4:
lambdas=[1.0, 100.0]

for lmbd in lambdas:
    RR = RidgeRegression()
    RR.initialize(lmbd,eta)
    RR.calc_w(X_train,Y_train)    
    RR.gradientDesc(X_train,Y_train, eta, epsilon,n_iter, w_init)
    RR.predict(b,X_train)
    result = RR.result()
    
    #print(result.w_calc)
    #print (result.w)
    RR.plot()
    
#def question5():       
#%%%
#Question5:
       
X_test=np.random.uniform(low=-5, high=5, size=100) 
Y_test=np.sin(X_test)+0.3*X_test-1

lambdas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100] 
losses=[]
for lmbd in lambdas: 
    RR = RidgeRegression()
    RR.initialize(lmbd,eta)
    RR.calc_w(X_train,Y_train)    
    RR.gradientDesc(X_train,Y_train, eta, epsilon,n_iter, w_init)
    RR.predict(b,X_test)
    l=RR.lossfunc(Y_test)
    losses=losses.append(l)
    result = RR.result()
print(losses)   
    
    
#if __name__ == '__main__':
    #question2()
    #question3()
    #question4()
    #question5()
    
 #%%

#Question6:  
 
 
 
 
 
 
 
 