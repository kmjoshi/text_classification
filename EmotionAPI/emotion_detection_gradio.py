import os
import gradio
# from flask import Flask, request, render_template, jsonify
from sklearn.externals import joblib

# https://stackoverflow.com/questions/12277933/send-data-from-a-textbox-into-flask


def text_preprocessing(s):
    import re
    import nltk

    # initialize tokenizer
    tweet_tokens = nltk.tokenize.TweetTokenizer(strip_handles=True)
    stem = nltk.stem.snowball.PorterStemmer()

    # https://regexr.com/
    s = re.sub(
        "(@[A-Za-z0-9]+)|([^0-9A-Za-z!? \t])|(\w+:\/\/\S+)|(http\S+)|(www\.\S+)", "", s)
    s = re.sub(r'\b\d+', '_', s)  # TODO: replace multiple digits w/ a _
    s = stem.stem(s)
    s = tweet_tokens.tokenize(s)

    return s


# get file path
dir_path = os.path.dirname(os.path.realpath(__file__))

model_name = dir_path + '/best_model_mlp.pkl'
model = joblib.load(model_name)

preprocessor_name = dir_path + '/Tfidf_preprocessor.pkl'
preprocessor = joblib.load(preprocessor_name)


def predict(s):
    # s: single str object

    pred = model.predict_proba(
        preprocessor.transform(
            [text_preprocessing(s)]))

    labels = ['anger', 'disgust', 'fear', 'joy', 'sadness']

    print(pred)
    output = dict(zip(labels, [round(x*100, 2) for x in pred[0]]))

    return output

# "EmotionAPI is a small 2-layer NN text-classification model trained on a
# combination of annotated tweets and ISEAR survey data"


# gradio setup
# model_type: ['sklearn', 'keras', 'pytorch', 'pyfunc']
out = gradio.outputs.Label(num_top_classes=5)
io = gradio.Interface(inputs="text", outputs="label",
                      model_type="pyfunc", model=predict)
io.launch()
