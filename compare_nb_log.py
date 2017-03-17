import sys
import os
import pickle
import re
from numpy import *
import tensorflow as tf
from logistic import part4
from naivebayes import part2
import pickle as cPickle


def compare_nb_log():

    try:
        snapshot = pickle.load(open("log_params.pkl", "rb"))
        W_log = snapshot["W"]
        vocab = snapshot["vocabulary"]
    except (OSError, IOError) as e:
        print("Running Part4 logistic regression to get Thetas...")
        W_log, vocab = part4(verbatim=False)
    try:
        snapshot = pickle.load(open("bayes_params.pkl", "rb"))
        wordDict_bayes = snapshot["bayes_wordDict"]
    except (OSError, IOError) as e:
        print("Running Part2 naive bayes to get Thetas...")
        wordDict_bayes = part2(verbatim=False)
        
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
    log_thetas = zeros(vocab_size)
    log_thetas[:] = W_log[:, 0] - W_log[:, 1] #... > 0, thres = 0
    sorted_inds = argsort(log_thetas)
    sorted = log_thetas[sorted_inds]
    
    #Greatest 10 Positive words should show up too...
    print("Printing top 100 thetas for logistic regression: \n")
    for i in range((vocab_size-100),vocab_size):
        print("Theta " + str(i+1) + " " + str(sorted[i]) + " Pos Word: " + vocab[sorted_inds[i]])
    
    print('\n')
    #NAIVE BAYES
    #1. grab the conditional prob from the dicts
    #2. log(p_ai / (count_class + k)) is one theta part - call it theta_(+/-)
    #3. for each theta, its actually theta_+ - theta_-
    #4. use those in the calculations!
    #get p_ai / (count_class + k), don't forget to perform the log on it!!!
    
    bayes_thetas = []
    for word in wordDict_bayes:
        bayes_thetas.append((wordDict_bayes[word][0] - wordDict_bayes[word][1], word))
    bayes_thetas.sort()
    
    print("Printing top 100 thetas for logistic regression: \n")
    i = 1
    for theta in bayes_thetas[-100:]:
        print("Theta " + str(i) + " " + str(theta[0]) + \
        " Pos Word: " + theta[1])
        i += 1
 
if __name__ == "__main__":
    print("================== RUNNING PART 6 ===================")
    compare_nb_log()
    
    
    
    
    
    
    