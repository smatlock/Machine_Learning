#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

## build model ##
from sklearn.naive_bayes import GaussianNB #import sklearn
clf = GaussianNB() #create classifier
t0 = time() #time training algorithm
fit = clf.fit(features_train, labels_train) #fit model
print "training time:", round(time()-t0, 3), "s" #time it takes to train
t1 = time() #time prediction algorithm
pred = clf.predict(features_test) #predict test features
print "prediction time:", round(time()-t1,3), "s" #time it takes to predict

## check accuracy ##
from sklearn.metrics import accuracy_score # import accuracy score (prediction, test labels)
accuracy = accuracy_score(pred, labels_test)
#print(accuracy)


#########################################################


