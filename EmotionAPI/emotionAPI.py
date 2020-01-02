from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib

load_dotenv()
# https://stackoverflow.com/questions/12277933/send-data-from-a-textbox-into-flask
# https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166

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


app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictEmotion(Resource):
    def __init__(self):
        self.model, self.preprocessor, self.labels = load_model()

    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        pred = self.model.predict_proba(
            self.preprocessor.transform(
                [text_preprocessing(user_query)]))

        output = dict(zip(self.labels, [round(x*100, 2) for x in pred[0]]))

        return output

class PredictEmotionBiLSTM(Resource):
    def __init__(self):
        self.model, self.preprocessor, self.labels = load_model()

    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        pred = self.model.predict_proba(
            self.preprocessor.transform(
                [text_preprocessing(user_query)]))

        output = dict(zip(self.labels, [round(x*100, 2) for x in pred[0]]))

        return output

api.add_resource(PredictEmotion, '/')
# api.add_resource(PredictEmotionBiLSTM, '/bilstm')

if __name__ == '__main__':
    app.run(debug=True)
