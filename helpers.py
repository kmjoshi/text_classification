def identity_tokenizer(text):
    """ return self

    """
    return text


def load_data(filename):
    """ loads data from .pkl files

    Parameters:
    filename (str): name of file to be unpickled

    Returns:
    X_train, X_test, Y_train, Y_test, labels

    """

    import pickle
    import numpy as np

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


def keras_preprocessing(text, tokenizer, max_len_seq=191):
    """ preprocess text for keras embedding block

    Parameters:
    text (list(str)): list of strings
    tokenizer (Tokenizer object)
    max_len_seq (int)

    Returns:
    temp (list(str)): processed and tokenized list of strings
    """
    from keras.preprocessing.sequence import pad_sequences
    temp = tokenizer.texts_to_sequences(text)
    temp = pad_sequences(temp, maxlen=max_len_seq, padding='post',
                         truncating='post')
    return temp


def custom_class_score(X_test, Y_test, clf, if_keras=False, tokenizer=None):
    """ Get a custom score by class that is outputted as a table

    Parameters: 
    X_test (np.array) (max_len, num_samples): unprocessed text
    Y_test (list(list)) (num_samples, num_classes):
    clf (sklearn clf or keras model):
    if_keras (Boolean): to indicate which methods to use
    tokenizer (Tokenizer object): provide tokenizer for preprocessing text

    Returns:
    n+6 x n+1 table where n is the number of classes
    n x n populated by the number of values misclassified as another
    added row for totals
    added columns for totals | precision | recall | f1 | pct | mse

    """

    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

    # need to be defensively coded
    if if_keras:
        # clf is a keras model
        probs = clf.predict(keras_preprocessing(X_test, tokenizer))
    else:
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
        precs[i] = precision_score(
            np.array(Y_test)[:, i], np.array(preds)[:, i])
        recs[i] = recall_score(np.array(Y_test)[:, i], np.array(preds)[:, i])
        f1[i] = f1_score(np.array(Y_test)[:, i], np.array(preds)[:, i])
    # get mis-labels
    mislabels = np.zeros((5, 5))
    for i in range(0, len(label_bool)):
        if label_bool[i] == False:
            mislabels[preds.argmax(axis=1)[i], Y_test.argmax(axis=1)[i]] += 1

    return mse, pct, precs, recs, f1, mislabels


def get_mislabels(X_test, Y_test, labels, clf, file, if_keras=False, tokenizer=None):
    """ save a table of misclassified labels

    Parameters: 
    X_test (np.array) (max_len, num_samples): unprocessed text
    Y_test (list(list)) (num_samples, num_classes):
    labels (list(str)): list of class labels
    clf (sklearn clf or keras model):
    if_keras (Boolean): to indicate which methods to use
    tokenizer (Tokenizer object): provide tokenizer for preprocessing text

    Returns:
    saves:
        n+6 x n+1 table where n is the number of classes
        n x n populated by the number of values misclassified as another
        added row for totals
        added columns for totals | precision | recall | f1 | pct | mse

    """

    import pandas as pd
    mse, pct, precs, recs, f1, mislabels = custom_class_score(
        X_test, Y_test, clf, if_keras=if_keras, tokenizer=tokenizer)

    df_mislabels = pd.DataFrame(mislabels, index=labels, columns=labels)
    df_mislabels['predicted_total'] = df_mislabels.sum(axis=1)
    df_mislabels['precision'] = precs
    df_mislabels['recall'] = recs
    df_mislabels['f1'] = recs
    df_mislabels['pct'] = pct
    df_mislabels['mse'] = mse
    df_mislabels.loc['missed_total'] = df_mislabels.sum(axis=0)

    df_mislabels.to_csv(file + '_mislabels.csv')


def create_tokenizer(infile='./split_text_data.pkl', outfile='embedding_matrix.pkl'):
    """ create and save a tokenizer on our dataset

    Parameters:
    infile (str): filename to load dict of data
    outfile (str): filename to save embedding matrix

    Returns:
    saved embedding matrix as outfile.pkl 
    """

    from keras.preprocessing.text import Tokenizer, text_to_word_sequence
    import numpy as np

    X_train, _, X_test, _, labels = load_data(infile)

    # initialize and fit tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_test)

    # load fastText embeddings model
    import fastText
    model_name = '../datasets/fastText_embeddings/cc.en.300.bin'
    ft_embeddings = fastText.load_model(model_name)

    # what
    nb_words = len(tokenizer.word_index)
    embedding_matrix = np.zeros((nb_words, 300))
    for word, i in tokenizer.word_index.items():
        embedding_matrix[i-1] = ft_embeddings.get_word_vector(word)

    import pickle
    with open(outfile, 'wb') as f:
        pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
