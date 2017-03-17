import sys
import os
import pickle
import re
from numpy import *
import tensorflow as tf
import pickle as cPickle

run_P1 = False
run_P2 = True


subdirs = [ ]


# You will work with the Cornell Movie Reviews dataset. You will want to 
# it to convert all the words to lowercase, and to get rid of punctuation. 
# Download the polarity dataset , and randomly split it into a test, 
# a validation, and a training set. (Use e.g. 200 movies for testing, 
# and 200 movies for validation.)


def parseFile(path, filename):
    wordlist = []
    with open(path + "/" + filename,"r") as f:
        for line in f:
            line_lower = line.lower()
            for word in line.split():
                word = ''.join([i for i in word if i.isalpha()])
                if word.isalnum(): #final check (empty string, etc)...
                    wordlist.append(word)
    return wordlist

#For entire , count the number of occurences, store in wordDict
# change so that occurrence counts ONLY once for each sample

def parseSet(set, set_labels):
    posDict = {}
    negDict = {}
    for i in range(len(set)):
        sample = set[i]
        sample_length = len(sample)
        for j in range(sample_length):
            word = sample[j]
            if word not in sample[:j]:#no occurrence yet
                if set_labels[i] == 1: #positve
                    if word not in posDict:
                        posDict[word] = float(1) #/ sample_length #NORMALIZE
                    else:
                        posDict[word] += float(1) #/ sample_length
                else: #negative
                    if word not in negDict:
                        negDict[word] = float(1) #/ sample_length
                    else:
                        negDict[word] += float(1) #/ sample_length
    return posDict, negDict
    

def prepare_data():
    path = 'txt_sentoken'
    reviews = []
    review_labels = []
    
    pos_filelist = os.listdir('txt_sentoken/pos') 
    for filename in pos_filelist:
        word_list = parseFile(path+'/pos', filename)
        reviews.append(word_list)
        review_labels.append(1)
        
    neg_filelist = os.listdir('txt_sentoken/neg') 
    for filename in neg_filelist:
        word_list = parseFile(path+'/neg', filename)
        reviews.append(word_list)
        review_labels.append(0)
    return reviews, review_labels

    
def generate_random_set(n_train=1600, n_val=200, n_test=200):
    reviews, review_labels = prepare_data()
    #Shuffle
    review_and_labels = list(zip(reviews, review_labels))
    random.shuffle(review_and_labels)
    reviews, review_labels = zip(*review_and_labels)
    
    num_samples = len(review_labels)
    
    train_set = []
    valid_set = []
    test_set = []

    train_l = [] #labels (0 or 1)
    valid_l = []
    test_l = []
    total_num = n_train+n_val+n_test
    
    if num_samples >= total_num:
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



## need modification 
# Give 3 examples of specific keywords that may be useful, 
# together with statistics on how often they appear in positive and negative reviews
#Part 1 I think just need to be finding the most frequenty occuring words, 
#the fancy ratio stuff comes later in Part 3

#Give 3 examples of specific keywords that may be useful, 
#together with statistics on how often they appear in positive and 
#negative reviews

def part1():
    num_pos = len([f for f in os.listdir('txt_sentoken/pos')])
    num_neg = len([f for f in os.listdir('txt_sentoken/neg')])
    print("The number of positive reviews is {0}".format(num_pos))
    print("The number of negative reviews is {0}".format(num_neg))
    reviews, review_labels = prepare_data()
    posDict, negDict = parseSet(reviews, review_labels)
    # newdict for the combined effect of pos and neg
    newdict = { k: posDict.get(k, 0) - negDict.get(k, 0) for k in set(posDict) | set(negDict) }
    keywords_sorted = sorted(newdict, key=newdict.get, reverse=False)
    keyword_neg = keywords_sorted[:3]
    keyword_pos = keywords_sorted[-3:]
    stub = "Word: {0} Positive count: {1} Negative count: {2}"
    print("3 keywords useful for Positive reviews:")
    for i in range(3):
        word = keyword_pos[i]
        print(stub.format(word, posDict[word], negDict[word]))
    print("3 keywords useful for Negative reviews:")
    for i in range(3):
        word = keyword_neg[i]
        print(stub.format(word, posDict[word], negDict[word]))
    
    

## THIS NEEDS TO BE LOG ODDS
## SEE this link
# http://pages.cs.wisc.edu/~jerryzhu/cs769/nb.pdf
# Naive Bayes as a Linear Classifier

# The log odds ratio will be greater than one for features which cause 
# belief in the Pos Class to be greater relative to Neg Class
# The features that have the greatest impact at classification time are those 
# with both a high probability (because they appear often in the data) 
# and a high odds ratio (because they strongly bias one label versus another).   

#List the 10 words that most strongly predict that the review is positive, 
#and the 10 words that most strongly predict that the review is negative. 
#State how you obtained those in terms of the the conditional probabilities 
#used in the Naive Bayes algorithm.
def part3(wordDict):
    logOdds = []
    
    for word in wordDict:
        logOdds.append((wordDict[word][0] - wordDict[word][1], word))
    logOdds.sort()
    
    vocab_size = len(logOdds)
    print("10 most strongly Positive word predictions:")
    print([word for val, word in logOdds[(vocab_size-10):]])
    print("10 most strongly Negative word predictions:")
    print([word for val, word in logOdds[:10]])
    
def get_cond_ai(count_ai, count_class, m, k):
    return log(float(count_ai + m*k)) - log(float(count_class + k))
        
def get_condProbabilites(posDict, negDict, count_p, count_n, m, k):
    wordDict = { key: [get_cond_ai(posDict.get(key, 0.0), count_p, m, k), \
    get_cond_ai(negDict.get(key, 0.0), count_n, m, k)] \
    for key in set(posDict) | set(negDict) }
    return wordDict
        

def part2(verbatim=True):
    train_set, train_l, valid_set, valid_l, test_set, test_l = generate_random_set()
    
    #Training Data
    posDict, negDict = parseSet(train_set, train_l)
    
    #class prior probabilities (positive, negative)
    train_pos_count = sum(train_l)
    prob_p = float(train_pos_count) / len(train_l)
    prob_n = float((len(train_l) - train_pos_count)) / len(train_l)
    count_class_p = sum([posDict[item] for item in posDict])
    count_class_n = sum([negDict[item] for item in negDict])
    
    #tune parameter m...
    
    # CHANGE THE GRID VALUES.....!!!!
    m_grid = arange(1, 2, 0.2)
    k_grid = arange(0.5, 10, 0.5)
    
    
    m, k = train_bayes(m_grid, k_grid, posDict, negDict, prob_p, count_class_p, prob_n, count_class_n, valid_set, valid_l)

    wordDict = get_condProbabilites(posDict, negDict, count_class_p, count_class_n, m, k)
    
    if verbatim:
        default_pos = log(float(m*k)) - log(float(count_class_p + k))
        default_neg = log(float(m*k)) - log(float(count_class_n + k))
        
        print("Final tuned parameters m=" + str(m) + " k=" + str(k))
        train_perf = get_accuracy(train_set, train_l, wordDict, prob_p, prob_n, \
        default_pos, default_neg)
        print("Performance on training set: " + str(100.0*train_perf))
        valid_perf = get_accuracy(valid_set, valid_l, wordDict, prob_p, prob_n, \
        default_pos, default_neg)
        print("Performance on validation set: " + str(100.0*valid_perf))
        test_perf = get_accuracy(test_set, test_l, wordDict, prob_p, prob_n, \
        default_pos, default_neg)
        print("Performance on test set: " + str(100.0*test_perf))
        
    snapshot = {}
    snapshot["bayes_wordDict"] = wordDict
    cPickle.dump(snapshot, open("bayes_params.pkl", "wb"))
    
    return wordDict
    
# Classify input x with thetas already trained (wordDict, ...)    
def get_accuracy(x, t, wordDict, prob_p, prob_n, default_pos, default_neg):
    hits = 0
    for i in range(len(x)):
        sample = x[i]
        
        # Compute the negative and positive probabilities
        #Positive
        positive_prediction = 0
        for word in set(sample):
            if word in wordDict:
                positive_prediction += wordDict[word][0] #log prob of P(ai=1 | class)
            else:
                positive_prediction += default_pos
        positive_prediction += log(prob_p)
        #Negative
        negative_prediction = 0
        for word in set(sample):
            if word in wordDict:
                negative_prediction += wordDict[word][1] #log prob of P(ai=1 | class)
            else:
                negative_prediction += default_neg
        negative_prediction += log(prob_n)

        prediction = 0
        # We assign a classification based on which probability is greater
        if positive_prediction > negative_prediction:
            prediction = 1
        if prediction == t[i]:
            hits += 1
    return float(hits) / len(t)   


########### FUNCTIONS FOR TRAINING ##########################################

def train_bayes(m_grid, k_grid, posDict, negDict, prob_p, count_p, prob_n, count_n, valid_set, valid_l):
    best_accuracy = -1
    for m in m_grid: # Smoothing parameter tuning loops
        print("m: " + str(m))
        for k in k_grid:
            print("k: " + str(k))
            accuracy = classify_bayes(valid_set, valid_l, posDict, negDict, \
            prob_p, count_p, prob_n, count_n, m, k)
            print(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (m, k)
    return best_params

def make_class_prediction(sample, class_worddict, class_prob, count_class, m, k=1):
    prediction = 0
    # remove duplicates
    sample_words = set(sample)
    #compute P(a1, ...an | class)
    for word in sample_words:
        if word in class_worddict:
            p_ai = float(class_worddict[word] + m*k)
        else:
            p_ai = float(m*k)
        prediction += log(float(p_ai)) - log(float(count_class + k))
    #prediction = exp(prediction)
    return prediction + log(class_prob)
    
def classify_bayes(x, t, posDict, negDict, prob_p, p_count, prob_n, n_count, m, k=1):
    hits = 0
    for i in range(len(x)):
        sample = x[i]
        # Compute the negative and positive probabilities
        positive_prediction = make_class_prediction(sample, posDict, prob_p, p_count, m, k)
        negative_prediction = make_class_prediction(sample, negDict, prob_n, n_count, m, k)
        
        prediction = 0
        # We assign a classification based on which probability is greater
        if positive_prediction > negative_prediction:
            prediction = 1
        if prediction == t[i]:
            hits += 1
    return float(hits) / len(t)
    
    
    

    
    
    
    
    
    
if __name__ == "__main__":
    random.seed(0)
    print("================== RUNNING PART 1 ===================")
    part1()
    print("================== RUNNING PART 2 ===================")
    wordDict = part2(verbatim=True)
    print("================== RUNNING PART 3 ===================")
    part3(wordDict)




