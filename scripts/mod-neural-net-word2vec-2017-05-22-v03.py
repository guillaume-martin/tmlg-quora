# -*- coding: utf-8 -*-
"""
Created on Thu May 22 2017

@author: guillaume

@version: 03
@changes
"""

#%% Import packages

import re
import csv
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split


#%% Set parameters

# directories
DATA_DIR = 'D:\\DataScience\\tmlg-quora\\data\\'
SUBMISSIONS_DIR = 'D:\\DataScience\\tmlg-quora\\submissions\\'
TRAINING_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
EMBEDDING_FILE = 'D:\\DataScience\\tmlg-quora\\data\\GoogleNews-vectors-negative300.bin'

# parameters
# Preporcessing paramaters
STOPWORDS = False
STEM = False
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 25

# Model validation parameters
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Neural network parameters
NB_EPOCHS = 200
DROPOUT_RATE = 0
DENSE_REG = 0.1
RNG_SEED = 13

STAMP = 'nn_maxseq%d_dreg%1.3f_dropout%1.2f_stopwd%s_stem%s' % (MAX_SEQUENCE_LENGTH, DENSE_REG, DROPOUT_RATE, STOPWORDS, STEM)
MODEL_WEIGHTS_FILE = STAMP + '.h5'

#%% Create an index of word vectors

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)


#%% Process text

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    ''' clean up questions
    Parameters
    ----------
    text : string
        The text of the question to be processed
    
    remove_stopwords : boolean
        True if stopwords need to be removed. False otherwise

    stem_word : boolean
        True if words need to be reduced to their stem. False otherwier

    Returns
    -------
    text : string
        
    '''
    # convert text to lowercase and split
    text = text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words('english'))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"how's","how is", text)
    text = re.sub(r"who's","who is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text


#%% Load the data

# import the training data
print('Importing training data')
train_question1 = []
train_question2 = []
train_labels = []
with open(DATA_DIR + TRAINING_DATA_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        train_question1.append(text_to_wordlist(row['question1'],STOPWORDS,STEM))
        train_question2.append(text_to_wordlist(row['question2'],STOPWORDS,STEM))
        train_labels.append(row['is_duplicate'])

print('Found %s question pairs in train.csv' % len(train_question1))

# import the test data
print('Importing test data')
test_question1 = []
test_question2 = []
test_ids = []
with open(DATA_DIR + TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        test_question1.append(text_to_wordlist(row['question1'],STOPWORDS,STEM))
        test_question2.append(text_to_wordlist(row['question2'],STOPWORDS,STEM))
        test_ids.append(row['test_id'])

print('Found %s question pairs in test.csv' % len(test_question1))


#%% Tokenize the questions

print('Training the tokenizer on all questions')
questions = train_question1 + train_question2 + test_question1 + test_question2
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)

print('Tokenizing questions')
train_q1_sequence = tokenizer.texts_to_sequences(train_question1)
train_q2_sequence = tokenizer.texts_to_sequences(train_question2)
test_q1_sequence = tokenizer.texts_to_sequences(test_question1)
test_q2_sequence = tokenizer.texts_to_sequences(test_question2)

word_index = tokenizer.word_index
print('Words in index: %d' % len(word_index))
questions = None

#%% Add padding to sequences

train_q1_data = pad_sequences(train_q1_sequence, maxlen=MAX_SEQUENCE_LENGTH)
train_q2_data = pad_sequences(train_q2_sequence, maxlen=MAX_SEQUENCE_LENGTH)
test_q1_data = pad_sequences(test_q1_sequence, maxlen=MAX_SEQUENCE_LENGTH)
test_q2_data = pad_sequences(test_q2_sequence, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.array(train_labels, dtype='int')

print('Shape of question1 tensor: ', train_q1_data.shape)
print('Shape of question2 tensor: ', train_q2_data.shape)
print('Shape of label tensor: ', labels.shape)

#%% Generate embedding matrix

print ('Generating embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# export the matrix
np.save(DATA_DIR + "embedding-matrix_nbwords%d_dim%d" % (nb_words, EMBEDDING_DIM))

#%% Train/validation split

X = np.stack((train_q1_data, train_q2_data), axis=1)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)

Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]


#%% Define the model structure

# can add regulizer to Dence layers
# model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))

Q1 = Sequential()
Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
Q2 = Sequential()
Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
model = Sequential()
model.add(Merge([Q1, Q2], mode='concat'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(DENSE_REG)))
model.add(Dropout(DROPOUT_RATE))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(DENSE_REG)))
model.add(Dropout(DROPOUT_RATE))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(DENSE_REG)))
model.add(Dropout(DROPOUT_RATE))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(DENSE_REG)))
model.add(Dropout(DROPOUT_RATE))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

#%% Compile the model

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_chekpoint = ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)

callbacks = [early_stopping, model_chekpoint]

#%% Train the model

print('Starting training at', datetime.datetime.now())

t0 = time.time()

history = model.fit([Q1_train, Q2_train],
                    y_train,
                    nb_epoch=NB_EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,
                    callbacks=callbacks)

t1 = time.time()

print('Training ended at ', datetime.datetime.now())
print('Minutes elapsed: %f' % ((t1 - t0) / 60.))

#%% Review model performance

model.load_weights(MODEL_WEIGHTS_FILE)

# get the accuracy
metrics = model.evaluate([Q1_test, Q2_test], y_test)
print('\nloss  = {0:.4f}'.format(metrics[0]))
print('accuracy  = {0:.4f}'.format(metrics[1]))

# save the best score 
best_val_score = min(history.history['val_loss'])

# list all data in history
# print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy\n' + STAMP)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('D:\\DataScience\\tmlg-quora\\img\\accuracy_' + STAMP + '.png')

# clean the plot
plt.clf() 

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss\n' + STAMP)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('D:\\DataScience\\tmlg-quora\\img\\loss_' + STAMP + '.png')


#%% Make predictions

print('Start making predictions')

X = np.stack((test_q1_data, test_q2_data), axis=1)

Q1_test = X[:,0]
Q2_test = X[:,1]

# from keras.models import load_model
# model = load_model('trained-keras-2017-05-17-v01')

predictions = model.predict([Q1_test, Q2_test],
                            batch_size=32,
                            verbose=1)


#%% Save submission file

print('\nExporting submission file')

test_ids = np.array(test_ids, dtype='int64')
submission = pd.DataFrame({'test_id':test_ids,'is_duplicate':predictions.ravel()})

timestamp = '{:%Y-%m-%d-%H-%M}'.format(datetime.datetime.now())

submission.to_csv(SUBMISSIONS_DIR + '%.4f_'%(best_val_score) + STAMP + '.csv', index=False)

        