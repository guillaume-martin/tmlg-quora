"""
this is an attempt to boost a trained neural network using adaboost
"""


#%% Import packages
import os
import numpy as np
import pandas as pd


#%% Setup directories and parameters

# Find what os is running and set the 
# directory separator accordingly
if os.name == "nt":
    # We're on windows
    SEPARATOR = "\"
else:
    # We're in Unix
    SEPARATOR = "/"


DATA_DIR = ".." + SEPARATOR + "data" + SEPARATOR
SUBMISISON_DIR = ".." + SEPARATOR + "submissions" + SEPARATOR
TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"
EMBEDDING_FILE = DATA_DIR + "GoogleNews-vectors-negative300.bin"

EMBEDDING_MATRIX = ""
TRAINED_NETWORK = ""
