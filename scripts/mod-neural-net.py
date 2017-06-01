"""
    created 2017/05/23
    Guillaume Martin

"""

#%% Import packcages
import numpy as np
import os, csv
from os.path import exists
from nltk import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#%% Setup directories and parameters

# Directories
# set a different directory depending on OS running the script
if os.name == "nt":
    # We're in Windows
    SEPARATOR = "\\"
else:
    # We're in a unix OS
    SEPARATOR = "/"

DATA_DIR = ".." + SEPARATOR + "data" + SEPARATOR
TRAINING_DATA_FILE = "train.csv"
TEST_DATA_FILE =  "test.csv"
EMBEDDING_FILE = ""

# Text preprocessing parameters
STOPWORDS = False
STEM = False

# Word embedding parameters
MAX_NUMBER_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300

# Model parameters
VALIDATION_SPLIT = 0.1
NB_EPOCHS = 200
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.1
EARLY_STOP = 3
SEED = 13

# Saving parameters
MODEL_WEIGHT_FILE = "model_dropout%d_l2reg%d.h5" % (DROPOUT_RATE, L2_REGULARIZATION)
TRAINING_Q1_FILE = "q1_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TRAINING_Q2_FILE = "q2_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TRAINING_LABELS_FILE = "training_labels.npy"
TEST_Q1_FILE = "testq1_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TEST_Q2_FILE = "testq2_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TEST_IDS_FILE = "test_ids.npy"


#%% Text processing function

def clean_text(text, remove_stopwords=False, stem_words=False):
    """ clean the text 
    parameters
    ----------
    text: string
        The text that needs to be processed

    remove_stopwords: boolean
        True if stopwords need to be removed. False otherwise

    stem_words: boolean
        True if words need to be reduce to their stem. False otherwise

    returns
    -------
    a cleaned up version of the text
    """

    # convert to lower case
    text = text.lower()

    # clean the text

    # we split the text to get a list of words
    text = text.split()

    # remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if w in stops]

    # reduce to stem
    if stem_words:
        stemmer = SnowballStemmer("english")
        text = [stemmer.stem(word) for word in text]

    text = " ".join(text)
    
    return text


#%% Load the data

# If we find saved pre-processed data, we load it.
# Else, we load the origina training and test data and process it
if exists(DATA_DIR + TRAINING_Q1_FILE) and exists(DATA_DIR + TRAINING_Q2_FILE) and exists(DATA_DIR + TRAINING_LABELS_FILE) and exists(DATA_DIR + TEST_Q1_FILE) and exists(DATA_DIR + TEST_Q2_FILE) and exists(DATA_DIR + TEST_Q2_FILE) and exists(DATA_DIR + TEST_IDS_FILE):
    print("Loading pre-processed data.")
    train_q1_data = np.load(open(DATA_DIR + TRAINING_Q1_FILE), "rb")
    train_q2_data = np.load(open(DATA_DIR + TRAINING_Q2_FILE), "rb")
    train_labels = np.load(open(DATA_DIR + TRAINING_LABELS_FILE), "rb")
    test_q1_data = np.load(open(DATA_DIR + TEST_Q1_FILE), "rb")
    test_q2_data = np.load(open(DATA_DIR + TEST_Q2_FILE), "rb")
    test_ids = np.load(open(DATA_DIR + TEST_IDS_FILE), "rb")
else:
    print("Loading original data.")
    print("Importing training data.")
    train_question1 = []
    train_question2 = []
    train_labels = []
    with open(DATA_DIR + TRAINING_DATA_FILE, encoding="utf-8") as f:
        reader = csv.DictReaded(f, delimiter=",")
        for row in reader:
            train_question1.append(clean_text(row["question1"]))
            train_question2.append(clean_text(row["question2"]))
            train_labels.append(row["is_duplicate"])

    print("Found %s question pairs in %s." % (len(train_question1), TRAINING_DATA_FILE))

    print("Importing test data.")
    test_question1 = []
    test_question2 = []
    test_ids = []
    with open(DATA_DIR + TEST_DATA_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            test_question1.append(clean_text(row["question1"]))
            test_question2.append(clean_text(row["question2"]))
            test_ids.append(row["test_id"])

    # tokenize the questions
    print("Training the tokenizer.")
    tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS)
    tokenizer.fit_on_texts(train_question1 + train_question2 + test_question1 + test_question2)

    print("Tokenizing questions.")
    # When tokenizing, we create lists of questions as lists of word
    # indexes.
    train_q1_sequence = tokenizer.texts_to_sequences(train_question1)
    train_q2_sequence = tokenizer.texts_to_sequences(train_question2)
    test_q1_sequence = tokenizer.texts_to_sequences(test_question1)
    test_q2_sequence = tokenizer.texts_to_sequences(text_question2)

    # The word index is a dictionary like {word:index}
    word_index = tokenizer.word_index
    print("Words in index: %d" % len(word_index))

    # Padding. We add values to all questions list so that they have
    # the same length (NN inputs need to have same length)
    print("Padding questions sequences.")
    train_q1_data = pad_sequences(train_q1_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    train_q2_data = pad_sequences(train_q2_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    test_q1_data = pad_sequences(test_q1_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    test_q2_data = pad_sequences(test_q2_sequence, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of question1 tensor: ', train_q1_data.shape)
    print('Shape of question2 tensor: ', train_q2_data.shape)
    print('Shape of label tensor: ', labels.shape)

    # We save all the files so they can be used with models
    # that have the same parameters without going through the
    # entire pre-processing
    print("Saving the pre-processed data to files.")
    np.save(TRAINING_Q1_FILE, train_q1_data)
    np.save(TRAINING_Q2_FILE, train_q2_data)
    np.save(TRAINING_LABELS_FILE, train_labels)
    np.save(TEST_Q1_FILE, test_q1_data)
    np.save(TEST_Q2_FILE, test_q2_data)
    np.save(TEST_IDS_FILE, test_ids)


#%% Index word vectors

print("Indexing word vectors.")
w2v = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)


#%% Generate embedding matrix

print('Generating embedding matrix')
nb_words = min(MAX_NUMBER_WORDS, len(word_index))
embedding_matrix  np.zeros((nb_words + 1, EMBEDDING_DIM))
# We search for the vector of each word in the word_index.
# If we find it, we add it to the embedding matrix.
for word, i in word_index_items():
    if word in w2v.vocab:
        embedding_matrix[i] = w2v.word_vec(word)