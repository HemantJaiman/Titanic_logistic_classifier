# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 01:43:07 2018

@author: Hemant Jaiman
"""
#Data preprocessing

import numpy as np
import pandas as pd


alpha=0.000009
sol_list=[]
t = np.zeros(5)
error_sum_before=1

data = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

testing_f = test.drop(['PassengerId','Name','Ticket','Fare','Cabin','Embarked'],axis=1)
testing_l=[]
final_list=test.iloc[:,0]



training_f=data.drop(['PassengerId','Survived','Name','Ticket','Fare','Cabin','Embarked'],axis=1)
training_l=data['Survived']


training_f['Age']=training_f['Age'].fillna(training_f['Age'].mean())
testing_f['Age']=testing_f['Age'].fillna(testing_f['Age'].mean())

training_f.Sex = pd.Categorical.from_array(training_f.Sex).codes
testing_f.Sex = pd.Categorical.from_array(testing_f.Sex).codes


#Logistic Regression       
for itr in range(0,20000):

    
    features = training_f.as_matrix()

    hypo = 1/(1+np.exp(-(np.matmul(t.T,features.T))))
    
    
    log_hypo=np.log(hypo)
    log_hypo1=np.log(1-hypo)
    log_hypo=log_hypo.T
    log_hypo1=log_hypo1.T
    sum_cost=0
    for i in range(0,891):
        sum_cost=sum_cost+(training_l[i]*(log_hypo[i]))+((1-training_l[i])*(log_hypo1[i]))
    error_Sum=-(sum_cost)/len(training_f)
    
        
        
    # Gradient Descend
  
    for i in range(0,len(training_f)):
        for j in range(0,len(t)):
           
            t[j]=t[j]-alpha*((hypo[i]-training_l[i])*features[i,j])
            
    print('the value of iteration no '+ str(itr) +'is' +str(error_Sum))

    if (error_Sum > error_sum_before):
        break;
    
    error_sum_before=error_Sum
    
tx = np.matmul(t.T,testing_f.T)
prediction = 1/(1+np.exp(-(tx)))


for i in range(0,418):
    
    if (prediction[i] >= 0.5):
        sol_list.append(1)
        
    else:
        sol_list.append(0)
        
print(sol_list)      


final_out=list(zip(final_list,sol_list))
sub=pd.DataFrame(final_out)


sub.to_csv("logistic_out.csv",index=False)

