import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, coverage_error, log_loss, precision_score, recall_score, f1_score, make_scorer, confusion_matrix, precision_recall_fscore_support
from helpers import load_data, custom_class_score, get_mislabels

import fastText

def prep_data_fastText():
    # outputs two files for processing in fastText. Do not need to run this online
    
    # prep data for fastText
    labeled_data = pd.DataFrame(data=Y_train, columns=labels).idxmax(axis=1)
    X_train_fastText = X_train.copy()
    for i, _ in enumerate(X_train):
        X_train_fastText[i] = '__label__' + labeled_data[i] + ' ' + X_train[i]
    with open('X_train_fastText.txt', 'w') as f:
        for line in X_train_fastText:
            f.write(line + '\n')

    # prep data for testing fastText
    labeled_data = pd.DataFrame(data=Y_test, columns=labels).idxmax(axis=1)
    X_test_fastText = X_test.copy()
    for i, _ in enumerate(X_test):
        X_test_fastText[i] = '__label__' + labeled_data[i] + ' ' + X_test[i]
    with open('X_test_fastText.txt', 'w') as f:
        for line in X_test_fastText:
            f.write(line + '\n')

# load data
X_train, Y_train, X_test, Y_test, labels = load_data('./split_text_data.pkl')
weights = Y_train.sum(axis=0)/Y_train.shape[0]

# load embeddings
model_name = '../datasets/fastText_embeddings/cc.en.300.bin'
ft_embeddings = fastText.load_model(model_name)

# train fastText
alpha = fastText.train_supervised('./X_train_fastText_subset.txt', dim=300, verbose=10, pretrainedVectors='../datasets/fastText_embeddings/cc.en.300.vec')

# test fastText: outputs precision in command-line
print(alpha.test('X_test_fastText.txt'))

# or predict on the test set and calculate my own precision value?
# needs to be tested

# extract embeddings
# get only sentence level embeddings
X_train_embedded = [ft_embeddings.get_word_vector(x) for x in X_train]

# get word-level embeddings for CNN/RNN/LSTMs
X_train_word_emebeddings = []
for i in X_train:
    empty_list = []
    for j in i:
        x = ft_embeddings.get_word_vector(i)
    

