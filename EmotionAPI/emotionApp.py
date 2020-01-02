import streamlit as st
import requests
from sklearn.externals import joblib

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

@st.cache
def load_model():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model_name = dir_path + '/best_model_mlp.pkl'
    print(model_name)
    model = joblib.load(model_name)

    preprocessor_name = dir_path + '/Tfidf_preprocessor.pkl'
    preprocessor = joblib.load(preprocessor_name)

    labels = ['anger', 'disgust', 'fear', 'joy', 'sadness']

    return model, preprocessor, labels

model, preprocessor, labels = load_model()

def get_emotions(query):
    pred = model.predict_proba(
    preprocessor.transform(
        [text_preprocessing(query)]))

    output = dict(zip(labels, [round(x*100, 2) for x in pred[0]]))
    return output

st.title('Big-5 Emotion Classification')
st.write('This widget queries the model and returns response below. Each emotion is rated on a scale of 0-100.')

query = st.text_input('Enter any emotion-filled text here!', 'There are absolutely no emotions in this text whatsoever!')
# response = requests.get('http://localhost:5000', params={'query': query})
response = get_emotions(query)

# st.json(response.json())
st.json(response)

st.markdown('Find slides of how the model was designed [here](https://github.com/kmjoshi/text_classification/blob/master/Emotion_Classification.pdf)')