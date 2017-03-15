import sys
import os
import pickle
import re
from numpy import *

# You will work with the Cornell Movie Reviews dataset. You will want to 
# it to convert all the words to lowercase, and to get rid of punctuation. 
# Download the polarity dataset , and randomly split it into a test, 
# a validation, and a training set. (Use e.g. 200 movies for testing, 
# and 200 movies for validation.)


def parseFile(path, filename):
    wordlist = []
    with open(path + "/" + file,"r") as f:
        for line in f:
            line_lower = line.lower()
            for word in line.split():
                word = ''.join([i for i in word if i.isalpha()])
                if word.isalnum(): #final check (empty string, etc)...
                    wordlist.append(word)
  return wordlist

#For entire , count the number of occurences, store in wordDict
# change so that occurence counts ONLY once for each sample

def parseSet(set, set_labels):
    posDict = {}
    negDict = {}
    for i in range(len(set)):
        sample = set[i]
        sample_length = len(sample_length)
        for j in range(len(sample_length))
            word = sample[j]
            if word not in sample[:j] #didn't already occur
                if set_labels[i] == 1: #positve
                    if word not in posDict:
                        posDict[word] = 1 / sample_length #NORMALIZE
                    else:
                        posDict[word] += 1 / sample_length
                else: #negative
                    if word not in negDict:
                        negDict[word] = 1 / sample_length
                    else:
                        negDict[word] += 1 / sample_length
    return posDict, negDict
    

def partition_data(n_train=1600, n_val=200, n_test=200):
    path = 'txt_sentoken'
    reviews = []
    review_labels = []
    
    pos_filelist = os.listdir('txt_sentoken\pos') 
    for filename in pos_filelist:
        word_list = parseFile(path, filename)
        reviews.append(word_list)
        review_labels.append(1)
        
    neg_filelist = os.listdir('txt_sentoken\neg') 
    for filename in neg_filelist:
        word_list = parseFile(path, filename)
        reviews.append(word_list)
        review_labels.append(0)

    #Shuffle
    review_and_labels = list(zip(reviews, review_labels))
    random.shuffle(review_and_labels)
    reviews, review_labels = zip(*c)
    
    num_samples = len(review_labels)
    
    train_set = []
    valid_set = []
    test_set = []

    train_l = [] #labels (0 or 1)
    valid_l = []
    test_l = []
    total_num = n_train+n_val+n_test
    
    if num_imgs >= total_num:
        for i in range(0,n_train):
            train_set.append(reviews[i])
            train_l.append(review_labels[i])
        for i in range(n_train,n_train+n_val):
            valid_set.append(reviews[i])
            valid_l.append(review_labels[i])
        for i in range(n_train+n_val,n_train+n_val+n_test):
            test_set.append(reviews[i])
            test_l.append(review_labels[i])
    else:
        raise ValueError('Not enough data to produce sets, %d needed, %d found' %(total_num, num_imgs))
    return train_set, train_l, valid_set, valid_l, test_set, test_l


def part2():
    train_set, train_l, valid_set, valid_l, test_set, test_l = partition_data()
    
    #Training Data
    posDict, negDict = parseSet(train_set, train_l)
    #save these ?
    #pickle.dump(posDict, open('pos.pickle', 'wb'))
    #pickle.dump(negDict, open('neg.pickle', 'wb'))
    
    #class prior probabilities (positive, negative)
    train_pos_count = sum(train_l)
    prob_p = train_pos_count / len(train_l)
    prob_n = (len(train_l) - train_pos_count) / len(train_l)
    
    #tune parameter m...
    
    # CHANGE THE GRID VALUES.....!!!!
    m_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    k_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    m, k = train_bayes(m_grid, k_grid, posDict, negDict, prob_p, prob_n, valid_set, valid_l)
    
    print("Final tuned parameters m=" + m + " k=" + k)
    train_perf = classify_bayes(train_set, train_l, posDict, negDict, prob_p, prob_n, m, k)
    print("Performance on training set: " + 100.0*train_perf)
    
    test_perf = classify_bayes(test_set, test_l, posDict, negDict, prob_p, prob_n, m, k)
    print("Performance on test set: " + 100.0*test_perf)
    
    
def train_bayes(m_grid, k_grid, posDict, negDict, prob_p, prob_n, valid_set, valid_l):
    best_accuracy = -1
    for m in m_grid: # Smoothing parameter tuning loops
        for k in k_grid:
            accuracy = classify_bayes(valid_set, valid_l, posDict, negDict, prob_p, prob_n, m, k)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (m, k)
    return best_params


def make_class_prediction(sample, class_worddict, class_prob, m, k=1):
    prediction = 0
    # remove duplicates
    sample_words = set(sample)
    count_class = sum(class_worddict.values())
    #compute P(a1, ...an | class)
    for word in sample_words:
        #compute P(ai=1 | class)
        p_ai = (class_worddict[word] + m*k) / (count_class + k)
        prediction += log(p_ai)
    prediction = exp(prediction)
    return prediction * class_prob
    
    
def classify_bayes(x, t, posDict, negDict, prob_p, prob_n, m, k=1):
    hits = 0
    for i in range(len(x)):
        sample = x[i]
        # Compute the negative and positive probabilities
        positive_prediction = make_class_prediction(sample, posDict, prob_p, m, k)
        negative_prediction = make_class_prediction(sample, negDict, prob_n, m, k)
        
        prediction = 0
        # We assign a classification based on which probability is greater
        if positive_prediction > negative_prediction:
            prediction = 1
        if prediction == t[i]:
            hits += 1
    return hits / len(t)



    



    
  


    







  



