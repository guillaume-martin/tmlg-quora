"""
This is an attempt to identify the parts of text that need
to be cleaned in the questions
"""

#%% Import packages
import re
import csv
import nltk
from gensim.models import KeyedVectors

#%% Setup directories and parameters
DATA_DIR = "../data/"
TRAINING_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
EMBEDDING_FILE = '~/Dropbox/DataScience/Data/GoogleNews-vectors-negative300.bin'

MAX_NB_WORDS = 200000

#%% Data cleaning

# we do a first cleaning of the text
def text_to_wordlist(text):

    # convert text to lowercase and split
    text = text.lower().split()

    text = " ".join(text)

    # clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"'re", " are", text)
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

    # we now tokenize the question to get a list of words
    text = nltk.word_tokenize(text)

    return text


#%% Load the data

questions = []

# Import the training data
print('Importing training data')
with open(DATA_DIR + TRAINING_DATA_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        questions.append(text_to_wordlist(row['question1']))
        questions.append(text_to_wordlist(row['question2']))

# import the test data
print('Importing test data')

with open(DATA_DIR + TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        questions.append(text_to_wordlist(row['question1']))
        questions.append(text_to_wordlist(row['question2']))

print('Found %s questions' % len(questions))


#%% word2vec

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

#%%

missing_words = []
words_in_index = []

for sentence in questions:
    for word in sentence:
        if word not in word2vec.vocab and word not in missing_words:
            print(word)
            missing_words.append(word)
        else:
            words_in_index.append(word)