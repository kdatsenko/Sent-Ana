import sys
import os
import pickle
import re
from numpy import *
import tensorflow as tf
from logistic import part4
import pickle as cPickle


def compare_nb_log():

    try:
        snapshot = pickle.load(open("log_params.pkl", "rb"))
        W_log = snapshot["W"]
        vocab = snapshot["vocabulary"]
    except (OSError, IOError) as e:
        print("Running Part4 logistic regression to get Thetas...")
        W_log, vocab = part4(verbatim=False)
        
        #Piazza:
        #The values will not necessarily be similar, 
        #depending on how you regularize logistic regression/tune the parameters of Naive Bayes
        
    #get top 100 thetas
    # http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
    #Relationship to Logistic Regression
    #In the special case where K=2K=2, one can show that softmax regression 
    # reduces to logistic regression. 
    
    #http://www.cs.toronto.edu/~guerzhoy/411/lec/W03/maximum_likelihood.pdf
    #Logistic Regression: Decision Surface
    #hyperplane decision surface
    
    vocab_size = W_log.shape[0]
    log_thetas = zeros((vocab_size, 2))
    log_thetas[:, 0] = W_log[:, 0] - W_log[:, 1] #... > 0, thres = 0
    log_thetas[:, 1] = arange(0, vocab_size, 1)
    sorted = log_thetas[np.argsort(log_thetas[:, 0]), :]
    
    #Greatest 10 Positive words should show up too...
    print("Printing top 100 thetas for logistic regression: ")
    for i in range(100):
        print("Theta " + str(i+1) + " " + sorted[i, 0] + "Pos Word: " + vocab[sorted[i, 1]])
        
    #NAIVE BAYES
    #1. grab the conditional prob from the dicts
    #2. log(p_ai / (count_class + k)) is one theta part - call it theta_(+/-)
    #3. for each theta, its actually theta_+ - theta_-
    #4. use those in the calculations!
    #get p_ai / (count_class + k), don't forget to perform the log on it!!!
    
        
    
    
    
    
    
    
    
    
    
    