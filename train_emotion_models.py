import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, coverage_error, log_loss, precision_score, recall_score, f1_score, make_scorer 

# helpers

# pipeline
# text_processing() | remove empty values | Tfidf_Vectorizer()
def load_data(filename):
    import pickle
    
    # load data
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    X_train = data.get('X_train')
    X_dev = data.get('X_dev')
    X_test = data.get('X_test')
    Y_train = data.get('Y_train')
    Y_dev = data.get('Y_dev')
    Y_test = data.get('Y_test')
    labels = data.get('labels')

    from scipy.sparse import vstack
    try:    # TODO: exception is very specific and might not generalize well
        X_train = vstack((X_train, X_dev))
    except ValueError:
        X_train = np.hstack((X_train, X_dev))

    Y_train = np.vstack((Y_train, Y_dev))

    return X_train, Y_train, X_test, Y_test, labels

def custom_class_score(X_test, Y_test, clf):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer 

    # need to be defensively coded
    probs = clf.predict_proba(X_test)
    Y_test = np.array(Y_test)
    label_tots = Y_test.sum(axis=0)
    label_bool = probs.argmax(axis=1) == Y_test.argmax(axis=1)
    # mse: [0, inf]
    mse = np.square(probs - Y_test).sum(axis=0)/label_tots
    # pct: [0, 100]
    pct = label_bool.sum()/len(Y_test)
    # precision by class
    preds = np.zeros(probs.shape)
    for i, x in enumerate(probs.argmax(axis=1)):
        preds[i, x] = 1
    precs = np.zeros((5,))
    recs = np.zeros((5,))
    f1 = np.zeros((5,))
    for i in range(0, 5):
        precs[i] = precision_score(np.array(Y_test)[:,i], np.array(preds)[:,i])
        recs[i] = recall_score(np.array(Y_test)[:,i], np.array(preds)[:,i])    
        f1[i] = f1_score(np.array(Y_test)[:,i], np.array(preds)[:,i])    
    # get mis-labels
    mislabels = np.zeros((5,5))
    for i in range(0, len(label_bool)):
        if label_bool[i] == False:
            mislabels[preds.argmax(axis=1)[i], Y_test.argmax(axis=1)[i]] += 1
    
    return mse, pct, precs, recs, f1, mislabels

def get_mislabels(X_test, Y_test, labels, clf, file):
    # get custom scores by class and save in a .csv file

    import pandas as pd
    mse, pct, precs, recs, f1, mislabels = custom_class_score(X_test, Y_test, clf)

    df_mislabels = pd.DataFrame(mislabels, index=labels, columns=labels)
    df_mislabels['predicted_total'] = df_mislabels.sum(axis=1)
    df_mislabels['precision'] = precs
    df_mislabels['recall'] = recs
    df_mislabels['f1'] = recs
    df_mislabels['pct'] = pct
    df_mislabels['mse'] = mse
    df_mislabels.loc['missed_total'] = df_mislabels.sum(axis=0)

    df_mislabels.to_csv(file + '_mislabels.csv')

# load data
X_train, Y_train, X_test, Y_test, labels = load_data('./split_text_tokenized_data.pkl')
weights = Y_train.sum(axis=0)/Y_train.shape[0]

# define default params
n_estimators = 10
max_depth = None
min_samples_leaf = 5
random_state = 7
verbose = 2
hidden_layer_sizes = (10,10)
alpha = 0.0001
learning_rate = 'constant'
max_iter = 200

# define estimators
clf_RFC = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf, 
                            n_jobs=-1, random_state=random_state, 
                             verbose=verbose)
clf_LDA = LinearDiscriminantAnalysis()
clf_BNB = BernoulliNB()
clf_GNB = GaussianNB()
clf_MLP = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, 
                       learning_rate=learning_rate, random_state=random_state, 
                       verbose=verbose)

clf_dict = {'rfc': clf_RFC, 'lda': clf_LDA, 'bnb': clf_BNB, 'gnb': clf_GNB, 
       'mlp': clf_MLP}
# params_RFC = [{'n_estimators': [10, 100, 1000], 'max_depth': [10, 100, 1000]}]
params_RFC = [{'n_estimators': [100], 'max_depth': [1000]}]
params_LDA = [{}]
params_BNB = [{}]
params_GNB = [{}]
# params_MLP = [{'hidden_layer_sizes': [(10,), (100,), (10,10), (100,100)]}]
params_MLP = [{'hidden_layer_sizes': [(10, 10), (40, 40), (10, 10, 10), (40, 40, 40), (100, 100, 100)]}]
params_grid = {'rfc': params_RFC, 'lda': params_LDA, 'bnb': params_BNB, 
               'gnb': params_GNB, 'mlp': params_MLP}

scorers = {'log_loss': make_scorer(log_loss, needs_proba=True), 'coverage_error': make_scorer(coverage_error), 'precision': make_scorer(precision_score, average='weighted'), 'recall': make_scorer(recall_score, average='weighted'), 'f1': make_scorer(f1_score, average='weighted'), 'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False, needs_proba=True, multioutput=weights)}

estimator_str = 'gnb'
clf = clf_dict[estimator_str]
params = params_grid[estimator_str]
file = './emotion_' + estimator_str

if estimator_str in ['lda', 'gnb', 'bnb']:
    scorers = {k:scorers[k] for k in ['precision', 'recall', 'f1']}
    print(scorers)

cv = GridSearchCV(clf, n_jobs=1, param_grid=params, refit='f1', 
                     cv=5, verbose=10, scoring=scorers, return_train_score=True)
# assert(X_train.shape[0] == Y_train.shape[0])
print('X: ' + str(X_train.shape) + '\n' + 'Y: ' + str(Y_train.shape))
if estimator_str in ['lda', 'bnb', 'gnb']:
    X_train = X_train.toarray()
    X_test = X_test.toarray()
if estimator_str in ['bnb', 'gnb']:
    Y_train = pd.DataFrame(data=Y_train).idxmax(axis=1).values
    Y_test = pd.DataFrame(data=Y_test).idxmax(axis=1).values
cv.fit(X_train, Y_train)

# get and save results
regTable = pd.DataFrame(cv.cv_results_)
regTable.to_csv(file + '.csv', index=False)
cols = [x for x in regTable.columns if 'mean' in x]
regTable[cols + ['params']].to_csv(file + '_means.csv', index=False)

if estimator_str in ['rfc', 'mlp']:
    get_mislabels(X_test, Y_test, labels, cv.best_estimator_, file)
else: 
    print(cv.score(X_test, Y_test))

from sklearn.externals import joblib
joblib.dump(cv.best_estimator_, './best_model_' + estimator_str + '.pkl')
