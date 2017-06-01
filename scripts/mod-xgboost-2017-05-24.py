"""
    created 2017/05/23
    Guillaume Martin

    This is an attempt at using xgboost with handcrafted features

"""

#%% Import packcages

import pandas as pd
import numpy as np
import os, string
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import metrics

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
EMBEDDING_FILE = "/home/guillaume/Dropbox/DataScience/Data/GoogleNews-vectors-negative300.bin"

# Text preprocessing parameters
PUNCTUATION = True
STOPWORDS = False
STEM = False

# Word embedding parameters
MAX_NUMBER_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300

# Model parameters
TEST_SPLIT = 0.1

# Saving parameters
TRAINING_Q1_FILE = "q1_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TRAINING_Q2_FILE = "q2_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TRAINING_LABELS_FILE = "training_labels.npy"
TEST_Q1_FILE = "testq1_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TEST_Q2_FILE = "testq2_stpwd%s_stem%s_maxwd%d_maxseq%d_embdim%d.npy" % (STOPWORDS, STEM, MAX_NUMBER_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
TEST_IDS_FILE = "test_ids.npy"


#%% Define functions

def text_to_wordlist(text, remove_punctuation=False, remove_stopwords=False, stem_words=False):
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
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize sentences
    text = word_tokenize(text.strip())

    # remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if w in stops]

    # reduce to stem
    if stem_words:
        stemmer = SnowballStemmer("english")
        text = [stemmer.stem(word) for word in text]

    return text


def text_avg_vector(wordlist, word2vec, num_features):
    """ calculate the average vector of a list of words
        by averaging the vectors of all the words in the list

        parameters:
        -----------
        wordlist: list
            a list of words

        word2vec:
            a trained word2vec model

        num_Features: integer
            the number of dimensions of the vectors

        returns
        -------
        feature_vec: array
            the average vector of the wordlist
    """

    feature_vec = np.zeros((num_features))

    nwords = 0

    for word in wordlist:
        if word in word2vec.vocab:
            nwords += 1
            feature_vec = np.add(feature_vec, word2vec.word_vec(word))

    feature_vec = np.divide(feature_vec, nwords)

    return feature_vec


def get_cosine(vec1, vec2):
    """ calculate the cosine of 2 vectors
    parameters
    ----------
    vec1: array
        the first word vector

    vec2: array
        the second word vector

    returns
    -------
    the cosine of the 2 vectors
    """

    # calculate the dot product of vec1 and vec2
    dotproduct = np.dot(vec1, vec2)

    # calculate the denominaror
    lenvec1 = np.sqrt(vec1.dot(vec1))
    lenvec2 = np.sqrt(vec2.dot(vec2))
    denominator = lenvec1 * lenvec2

    if denominator == 0:
        return 0.0
    else:
        return float(dotproduct) / denominator


def get_shared_words_pct(q1, q2, remove_stopwords=False, stem_words=False):
    """ counts the number of words that are common to the 2 questions
    parameters
    ----------
    q1: list
        the list of words in question1

    q2: list
        the list of words in question2

    remove_stopwords: boolean
        True if stopwords have to be removed from list. False otherwise.

    stem_words: boolean
        True if words have to be reduces to their stem. False otherwise

    return
    ------
    shared: integer
        the number of words that are common to both questions

    shared_pct: float
        the percentage of words that are shared over the total of unique words
        in the two quetions
    """

    # remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        q1 = [w for w in q1 if w not in stops]
        q2 = [w for w in q2 if w not in stops]

    # reduce to stem
    if stem_words:
        stemmer = SnowballStemmer("english")
        q1 = [stemmer.stem(w) for w in q1]
        q2 = [stemmer.ste(w) for w in q2]

    # remove duplicate words
    set1 = set(q1)
    set2 = set(q2)

    # calculate the number of words that are common to the two sets
    shared = len(set1 & set2)

    # count the number of unique words in the two sets
    unique_words = list(set(q1 + q2))
    unique_count = len(unique_words)

    # calculate the percentage of shared words
    if unique_count != 0:
        shared_pct = (shared / unique_count) * 100
    else:
        shared_pct = 0

    return shared_pct


def dummify(df, column, drop=True):
    ''' add dummy variables columns to a dataframe
    parameters
    ----------
        df    dataframe
            the dataframe that need to be modified

        column    strin
            the name of the column to dummify

        drop    boolean (default=True)
            True to drop the original column

    return
    ------
        a dataframe with extra columns
    '''

    df_dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df,df_dummies], axis=1)
    if drop == True:
        df.drop([column], inplace=True, axis=1)
    return df
#%% Load the data

train_df = pd.read_csv(DATA_DIR + TRAINING_DATA_FILE)
# test_df = pd.read_csv(DATA_DIR + TEST_DATA_FILE)


#%% Get word lists

print("Generating word lists from questions.")
train_df["q1_wordlist"] = train_df["question1"].apply(lambda x: text_to_wordlist(str(x), PUNCTUATION, STOPWORDS, STEM))
train_df["q2_wordlist"] = train_df["question2"].apply(lambda x: text_to_wordlist(str(x), PUNCTUATION, STOPWORDS, STEM))

# test_df["q1_wordlist"] = test_df["question1"].apply(lambda x: text_to_wordlist(x, PUNCTUATION, STOPWORDS, STEM))
# test_df["d2_wordlist"] = test_df["question2"].apply(lambda x: text_to_wordlist(x, PUNCTUATION, STOPWORDS, STEM))

#%% Index word vectors

print("Indexing word vectors.")
w2v = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

#%% Add average vector to features

train_df["q1_avg_vec"] = train_df["q1_wordlist"].apply(lambda x: text_avg_vector(x, w2v, EMBEDDING_DIM))
train_df["q2_avg_vec"] = train_df["q2_wordlist"].apply(lambda x: text_avg_vector(x, w2v, EMBEDDING_DIM))

# test_df["q1_avg_vec"] = train_df["q1_wordlist"].apply(lambda x: text_avg_vector(x, w2v, EMBEDDING_DIM))
# test_df["q2_avg_vec"] = train_df["q2_wordlist"].apply(lambda x: text_avg_vector(x, w2v, EMBEDDING_DIM))

# free memory
w2v = None
#%% Get cosine similarity of the 2 questions average vectors

train_df["cosine"] = train_df.apply(lambda row: get_cosine(row["q1_avg_vec"], row["q2_avg_vec"]), axis=1)
# test_df["cosine"] = test_df.apply(lambda row: get_cosine(row["q1_avg_vec"], row["q2_avg_vec"]), axis=1)


#%% Get the distance of the 2 questions average vectors

train_df["distance"] = train_df.apply(lambda row: np.linalg.norm(row["q1_avg_vec"] - row["q2_avg_vec"]), axis=1)
# test_df["distance"] = test_df.apply(lambda row: np.linalg.norm(row["q1_avg_vec"] - row["q2_avg_vec"]), axis=1)


#%% Get the character count difference

train_df["char_diff"] = train_df.apply(lambda row: abs(len(str(row["question1"])) - len(str(row["question2"]))), axis=1)
train_df["char_diff_bins"] = pd.cut(train_df["char_diff"], 50)

#test_df["char_diff"] = df_test.apply(lambda row: abs(len(str(row["question1"])) - len(str(row["question2"]))), axis=1)
#test_df["char_diff_bin"] = pd.cut(test_df["char_diff"], 50)


#%% Get the word count difference

train_df["word_diff"] = train_df.apply(lambda row: abs(len(str(row["q1_wordlist"])) - len(str(row["q2_wordlist"]))), axis=1)
train_df["word_diff_bins"] = pd.cut(train_df["word_diff"], 25)

#test_df["word_diff"] = test_df.apply(lambda row: abs(len(str(row["q1_wordlist"])) - len(str(row["q2_wordlist"]))), axis=1)
#test_df["word_diff_bins"] = pd.cut(test_df["word_diff"], 25)


#%% Count common words

train_df["shared_pct"] = train_df.apply(lambda row: get_shared_words_pct(row["q1_wordlist"], row["q2_wordlist"], True), axis=1)
train_df["shared_pct_bins"] = pd.cut(train_df["shared_pct"], 10)

#test_df["shared_words"] = train_df.apply(lambda row: get_shared_words_count(test_df["q1_wordlist"], test_df["q2_wordlist"], True))


#%% Get the shared percentage

# We want to know if the number of shared words represent a large part of the
# questions.
# We calculate the percentage of shared words over the total of unique words in
# the two questions

#%% Split the data

features = ["cosine","distance", "char_diff", "shared_pct"]

X = np.array(train_df[features])
y = np.array(train_df["is_duplicate"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)


#%% Build the model

model = XGBClassifier()


#%% Fit the model
print("Fitting the model")
model.fit(X_train, y_train)

#%% Make predictions on validation set
print("Making predictions")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)


#%% Get metrics

# I want to see the learning curves to spot overfitting
#from utils.ModelEvaluation import draw_learning_curves
#draw_learning_curves(model, X, y)

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10,
                                                        train_sizes=np.linspace(.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.mean(test_scores, axis=1)

plt.figure()
plt.grid()

plt.fill_between(train_sizes,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1, color='r')
plt.fill_between(train_sizes,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1, color='g')

plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
         label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
         label='Cross-validation score')

plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()


# I want to see the recall and precision
print(metrics.classification_report(y_test, y_pred))

# show the confusion matrix
cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
print(cm)
labels = [0,1]
matrix = metrics.confusion_matrix(y_test, y_pred, labels)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(matrix)
#cax = ax.matshow(matrix)
plt.title('Confusion Matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show



# get the auc score
print("AUC = %.4f" & metrics.roc_auc_score(y_test, y_proba[:,1]))

