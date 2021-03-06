#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:12:43 2017

@author: lizhuoran
"""

import sys
import os
import pickle
import re
import numpy as np
import scipy
import tensorflow as tf
import pickle as cPickle
import matplotlib.pyplot as plt
import random
import operator


prepare_dataP7 = False
train_P7 = True
run_P8 = True
find_example = True

X_SIZE = 128*128
N_LABELS = 2


#We are supplying you with word2vec embeddings for a list of word. 
#Show that the word2vec embeddings work for figuring out whether t appears 
#together with w or not using Logistic Regression. Describe the experiment 
#you performed and its results.

#The ([context], target) pairs are thus ([I, never], have), ([have, been], never), 
#    ([never, to], been), ([been, the], to), ([to, Centauri], Alpha).
#
#We would like to predict the context from the word. The (input, output) pairs are 
#thus (I, have), (never, have), (have, never), (been, never), (never, been), 
#     (never, to), (been, to), (the, to), (to, Alpha), (Centauri, Alpha).
    


embeddings = np.load("embeddings.npz")["emb"]
indices = np.load("embeddings.npz")["word2ind"].flatten()[0]
vocab_size = embeddings.shape[0]
eb_size = embeddings.shape[1]  




def prepare_data():
    reviews = []
    wordDict = {} #maps words to indices
    vocabulary = {} # count occurrences of each word
    
    
    pos_filelist1 = os.listdir('txt_sentoken/pos')
    pos_filelist = random.sample(pos_filelist1, 500)
    for filename in pos_filelist:
        wordlist = []
        with open("txt_sentoken/pos/" + filename,"r") as f:
            for line in f:
                line_lower = line.lower()
                for word in line.split():
                    word = ''.join([i for i in word if i.isalpha()])
                    if word.isalnum(): #final check (empty string, etc)...
                        wordlist.append(word)
                        if word not in vocabulary.keys():
                            vocabulary[word] = 1
                        else:
                            vocabulary[word] += 1
                        
        reviews.append(wordlist)
    test_vocab = sorted(vocabulary, key=vocabulary.get, reverse=True)[170:173]
    common_vocab = sorted(vocabulary, key=vocabulary.get, reverse=True)[:800]
    
    filter_indices = {k: v for k, v in indices.items() if v in common_vocab}
        
    valid = list(filter_indices.values())
    contar = []
    d = {}
    
    # create a target-context dict for selected words(test_vocab)
    for wordlist in reviews:
        for i in range(1, len(wordlist)-1):
            target = wordlist[i]
            context1 = wordlist[i-1]
            context2 = wordlist[i+1]
            if target in test_vocab and context1 in valid and context2 in valid:
                if not target in d:
                    d[target] = [context1, context2]
                else:
                    d[target].append(context1)
                    d[target].append(context2)
    
    for tar in d:
        v = random.sample(valid, 500)
        t_ind = list(indices.keys())[list(indices.values()).index(tar)]
        for cont in d[tar]:
            cont_ind = list(indices.keys())[list(indices.values()).index(cont)]
            contar.append([t_ind, cont_ind, 1])
        for j in range(len(v)):
            v_ind = list(indices.keys())[list(indices.values()).index(v[j])]
            if v[j] not in d[tar]:
                contar.append([t_ind, v_ind, 0])  
    contar = np.asarray(contar)
    random.shuffle(contar)
        
    # process x, y for inputs    
    x_ind = contar[:, :2]
    x = []
    for pair in x_ind:
        eb0 = [embeddings[pair[0], :]]
        eb0 = np.asarray(eb0)
        eb1 = [embeddings[pair[1], :]]
        eb1 = np.asarray(eb1)
        mul = (eb0 * eb1.T).flatten()
        x.append(mul)
    x = np.asarray(x)
    y_p = contar [:, 2]
    y = np.zeros((x.shape[0], N_LABELS))
    for k in range(x.shape[0]):
        if y_p[k] == 0:
            y[k, 1] = 1
        else:
            y[k, 0] = 1
    return x, y
            


def generate_random_set(x, y, n_train=1600, n_val=180, n_test=180):
    
    num_samples = x.shape[0]
    shuffled_inds = np.random.permutation(num_samples)
    x = x[shuffled_inds]
    y = y[shuffled_inds]
    
    total_num = n_train+n_val+n_test
    
    train_x = np.zeros((n_train, X_SIZE))
    train_y = np.zeros((n_train, N_LABELS))
    valid_x = np.zeros((n_val, X_SIZE))
    valid_y = np.zeros((n_val, N_LABELS))
    test_x = np.zeros((n_test, X_SIZE))
    test_y = np.zeros((n_test, N_LABELS))
    
    if num_samples >= total_num:
        train_x[:, :] = x[0:n_train, :]
        train_y[:, :] = y[0:n_train, :]
        valid_x[:, :] = x[n_train:n_train+n_val, :]
        valid_y[:, :] = y[n_train:n_train+n_val, :]
        test_x[:, :] = x[n_train+n_val:n_train+n_val+n_test, :]
        test_y[:, :] = y[n_train+n_val:n_train+n_val+n_test, :]
    else:
        raise ValueError('Not enough data to produce sets, %d needed, %d found' %(total_num, num_imgs))
    return train_x, train_y, valid_x, valid_y, test_x, test_y
            

# input size okay?
# need hidden layer?

def logistic_train(train_x, train_y, valid_x, valid_y, test_x, test_y,  verbatim=True):
    
    
    #One fully connected layer 
    
    x = tf.placeholder(tf.float32, [None, X_SIZE])    
    
    W = tf.Variable(tf.random_normal([X_SIZE, N_LABELS], stddev=0.01))/10
    b = tf.Variable(tf.random_normal([N_LABELS], stddev=0.01))/10
    
    layer = tf.matmul(x, W)+b
    
    y = tf.nn.softmax(layer)
    y_ = tf.placeholder(tf.float32, [None, N_LABELS])

    lam = 0.0075
    decay_penalty =lam*tf.reduce_sum(tf.abs(W))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    alpha = 0.005
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(NLL)
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #for plotting
    train_error = []
    val_error = []
    test_error = []
    iter_xvals = []
    
    for i in range(1000):
        batch_xs = train_x 
        batch_ys = train_y
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        if verbatim:
            if i % 20 == 0:
                val_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
                train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
                test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
                
                iter_xvals.append(i)
                val_error.append(val_accuracy)
                test_error.append(test_accuracy)
                train_error.append(train_accuracy)
                if i % 100 == 0: #print out for user
                    print("i=",i)
                    print("Train:", train_accuracy)
                    print("Validation:", val_accuracy)
                    print("Test:", test_accuracy)
                    print("Penalty:", sess.run(decay_penalty))
    
    
    if verbatim:
        val_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
        train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
        test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        print("The final performance on the training set is: ", train_accuracy)
        print("The final validation set accuracy is: ", val_accuracy)
        print("The final performance on the test set is: ", test_accuracy)
        
        plt.ion()
        plt.figure(1)
        plt.plot(iter_xvals, train_error, label="Training Accuracy")
        plt.plot(iter_xvals, val_error, label="Validation Accuracy")
        plt.plot(iter_xvals, test_error, label="Test Accuracy")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy (%)')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=2, mode="expand", borderaxespad=0.)
        plt.savefig('part7_iteration_vs_accuracy')
        plt.show()
    
    

def part8(t):
    t_ind = list(indices.keys())[list(indices.values()).index(t)]
    t_em = embeddings[t_ind, :]
    similarity = {}
    for i in range(embeddings.shape[0]):
        s_em = embeddings[i, :]
        #dist = np.linalg.norm(s_em-t_em)
        dist1 = scipy.spatial.distance.cosine(s_em, t_em)
        similarity[indices[i]] = dist1
    most_similar = sorted(similarity.items(), key=operator.itemgetter(1))[1:11]
    print ("10 words that appear in similar contexts with " + t + ":")
    for tp in most_similar:
        print(tp)


# WTF a, b, c, d,  s.t. a+b = c+d
def find_examples(a, b, c):
    a_ind = list(indices.keys())[list(indices.values()).index(a)]
    b_ind = list(indices.keys())[list(indices.values()).index(b)]
    c_ind = list(indices.keys())[list(indices.values()).index(c)]
    combined = embeddings[a_ind, :] + embeddings[b_ind, :] - embeddings[c_ind, :]
    likely = {}
    for i in range(embeddings.shape[0]):
        s_em = embeddings[i, :]
        dist = scipy.spatial.distance.cosine(s_em, combined)
        likely[indices[i]] = dist
    most_likely = sorted(likely.items(), key=operator.itemgetter(1))[1:4]
    print (most_likely)
            
            
    
    
if __name__ == "__main__":
    random.seed(0)
    #if prepare_dataP7:
    x, y = prepare_data()
    train_x, train_y, valid_x, valid_y, test_x, test_y = generate_random_set(x, y)
    if train_P7:
        logistic_train(train_x, train_y, valid_x, valid_y, test_x, test_y)
    if run_P8:
        part8("story")
        print("===================================")
        part8("good")
    if find_example:
        find_examples("men", "woman", "women")
        find_examples("scenes", "movie", "movies")
    