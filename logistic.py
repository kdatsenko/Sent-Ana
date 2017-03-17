import sys
import os
import pickle
import re
from numpy import *
import tensorflow as tf
import pickle as cPickle


N_LABELS = 2

# Train a Logistic Regression model on the same dataset. 
#For a single movie review, For a single review r the input to the Logistic 
#Regression model will be a k-dimensional vector v, where v[k]=1v[k]=1 if 
#the k-th keyword appears in the review r. The set of keywords consists of 
#all the words that appear in all the reviews.
# 
# Plot the learning curves (performance vs. iteration) of the Logistic 
#Regression model. Describe how you selected the regularization parameter 
#(and describe the experiments you used to select it).
    

def prepare_data():
    reviews = []
    review_labels = []
    wordDict = {} #maps words to indices
    vocabulary = []
    index = 0
    
    pos_filelist = os.listdir('txt_sentoken/pos') 
    for filename in pos_filelist:
        wordlist = []
        with open("txt_sentoken/pos/" + filename,"r") as f:
            for line in f:
                line_lower = line.lower()
                for word in line.split():
                    word = ''.join([i for i in word if i.isalpha()])
                    if word.isalnum(): #final check (empty string, etc)...
                        wordlist.append(word)
                        if word not in wordDict:
                            wordDict[word] = index
                            vocabulary.append(word)
                            index += 1
        reviews.append(wordlist)
        review_labels.append(1)
        
    neg_filelist = os.listdir('txt_sentoken/neg') 
    for filename in neg_filelist:
        wordlist = []
        with open("txt_sentoken/neg/" + filename,"r") as f:
            for line in f:
                line_lower = line.lower()
                for word in line.split():
                    word = ''.join([i for i in word if i.isalpha()])
                    if word.isalnum(): #final check (empty string, etc)...
                        wordlist.append(word)
                        if word not in wordDict:
                            wordDict[word] = index
                            vocabulary.append(word)
                            index += 1
        reviews.append(wordlist)
        review_labels.append(0)
        
    VOCAB_SIZE = index
    num_samples = len(reviews)
    x = zeros((num_samples, VOCAB_SIZE))
    y = zeros((num_samples, N_LABELS))
    #Build vocab_size * num_samples matrix
    for i in range(num_samples):
        sample = reviews[i]
        #k-dimensional vector v, where v[k]=1v[k]=1 if 
        #the k-th keyword appears in the review r
        for word in sample:
            x[i, wordDict[word]] = 1
        # Each row is a one-hot vector of 2 classes (pos, neg)
        if review_labels[i] == 1:
            y[i, 0] = 1
        else:
            y[i, 1] = 1

    return VOCAB_SIZE, x, y, vocabulary

#1. Translate the entire set to those word vecotrs
#each row of the training set is a sample
#should be do batches? 
#2. Upgrade tens


def part4(verbatim=True):
    random.seed(0)
    vocab_size, x, y, vocabulary = prepare_data()
    tr, tr_y, v, v_y, te, te_y = generate_random_set(vocab_size, x, y)
    W = logistic_train(vocab_size, tr, tr_y, v, v_y, te, te_y, vocabulary, verbatim)
    return W, vocabulary
    
    
def generate_random_set(VOCAB_SIZE, x, y, n_train=1600, n_val=200, n_test=200):
    
    num_samples = x.shape[0]
    shuffled_inds = random.permutation(num_samples)
    x = x[shuffled_inds]
    y = y[shuffled_inds]
    
    total_num = n_train+n_val+n_test
    
    train_x = zeros((n_train, VOCAB_SIZE))
    train_y = zeros((n_train, N_LABELS))
    valid_x = zeros((n_val, VOCAB_SIZE))
    valid_y = zeros((n_val, N_LABELS))
    test_x = zeros((n_test, VOCAB_SIZE))
    test_y = zeros((n_test, N_LABELS))
    
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
    

def logistic_train(VOCAB_SIZE, train_x, train_y, valid_x, \
valid_y, test_x, test_y, vocabulary, verbatim=True):
    
    #One fully connected layer 
    
    x = tf.placeholder(tf.float32, [None, VOCAB_SIZE])    
    
    W = tf.Variable(tf.random_normal([VOCAB_SIZE, N_LABELS], stddev=0.01))/10
    b = tf.Variable(tf.random_normal([N_LABELS], stddev=0.01))/10
    
    layer = tf.matmul(x, W)+b
    
    y = tf.nn.softmax(layer)
    y_ = tf.placeholder(tf.float32, [None, N_LABELS])

    lam = 0.0085
    decay_penalty =lam*tf.reduce_sum(tf.square(W))
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
    
    for i in range(500):
        batch_xs = train_x #MINI BATCH METHOD???
        batch_ys = train_y#get_train_batch(M, 50)
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
        savefig('part4_iteration_vs_accuracy')
        plt.show()
    
    
    snapshot = {}
    W_sess = sess.run(W)
    W_params = W_sess.eval(session=sess)
    snapshot["W"] = W_params
    snapshot["vocabulary"] = vocabulary
    cPickle.dump(snapshot, open("log_params.pkl", "wb"))
    
    return W_params #Thetas (excluding theta_0 - the bias b...)
    

if __name__ == "__main__":
    print("================== RUNNING PART 4 ===================")
    part4(verbatim=True)