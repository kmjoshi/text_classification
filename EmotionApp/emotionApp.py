
import os
from flask import Flask, request, render_template, jsonify
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


app = Flask(__name__)

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


@app.route("/")
def my_form():
    return render_template('my-form.html')


@app.route("/", methods=['POST'])
def my_form_post():
    text = request.form['text']
    return jsonify(predict(text))


if __name__ == '__main__':
    app.run(debug=True)
