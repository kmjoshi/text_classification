import pandas as pd

# helpers

# define all preprocessing steps
# save this for the test set and API
def text_preprocessing(s):
	import re
	import nltk

	# initialize tokenizer
	tweet_tokens = nltk.tokenize.TweetTokenizer(strip_handles=True)
	stem = nltk.stem.snowball.PorterStemmer()

    # https://regexr.com/
	s = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z!? \t])|(\w+:\/\/\S+)|(http\S+)|(www\.\S+)", "", s)
	s = re.sub(r'\b\d+', '_', s)  # TODO: replace multiple digits w/ a _
	s = stem.stem(s)
	s = tweet_tokens.tokenize(s)
    
	return s

def load_data_from_file():
	# Load Twitter dataset and downsample 13-emotions to 5
	df = pd.read_csv('../datasets/text_emotion_0.csv')
	df.loc[(df.sentiment=='love') | (df.sentiment=='relief')
	        | (df.sentiment=='enthusiasm') | 
	        (df.sentiment=='fun') | (df.sentiment=='surprise'), 
	        'sentiment'] = 'happiness'
	df.loc[df.sentiment=='hate', 'sentiment'] = 'anger'
	df.loc[df.sentiment=='worry', 'sentiment'] = 'fear'
	df.loc[(df.sentiment=='empty') | (df.sentiment=='boredom'),
	        'sentiment'] = 'neutral'
	df.loc[df.sentiment=='happiness', 'sentiment'] = 'joy'
	df_tweets = df[['sentiment', 'content']].copy()

	# load ISEAR survey data
	df = pd.read_csv('../datasets/ISEAR_0/ISEAR_0.tsv', sep='\t')
	df.drop(df[(df.Field1=='guilt') | (df.Field1=='shame')].index, inplace=True)
	df_isear = df[['SIT', 'Field1']].rename(columns={'SIT': 'content', 'Field1': 'sentiment'})

	return df_tweets, df_isear

def apply_token_stems(df_tweets, df_isear):
	# tokenize
	df_tweets.content = df_tweets.content.apply(text_preprocessing)
	df_isear.content = df_isear.content.apply(text_preprocessing)

	return df_tweets, df_isear

def save_dataset(df_tweets, df_isear, filename, if_featurize=False, if_tokenize=False):
	# saves all the data as X & Y
	# featurized and tokenized according to flags
	import pickle
	from sklearn.externals import joblib 
	from helpers import identity_tokenizer

	if if_tokenize:
		df_tweets, df_isear = apply_token_stems(df_tweets, df_isear)

	# remove empty text
	df_tweets['len'] = df_tweets.content.apply(len)
	df_tweets = df_tweets[df_tweets.len > 0]
	df_isear['len'] = df_isear.content.apply(len)
	df_isear = df_isear[df_isear.len > 0]

	X = pd.concat([df_tweets.content, df_isear.content], axis=0).values
	Y = pd.concat([df_tweets.sentiment, df_isear.sentiment], axis=0).values
	
	if if_featurize:
		# Data Prep: bag of words
		from sklearn.feature_extraction.text import TfidfVectorizer
		features = TfidfVectorizer(lowercase=False, tokenizer=identity_tokenizer)  # keep it default?
		# fit the corpus to a bag of words model
		# save this for the test set and API
		features.fit(X)

		pickle.dump(features, open('Tfidf_preprocessor.pkl', 'wb'))
		# print('dumped')

		X = features.transform(X)

	print('X: ' + str(X.shape) + '\n' + 'Y: ' + str(Y.shape))
	data = {'X': X, 'Y': Y}
	with open(filename, 'wb') as f:
	    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

	print('Saved ' + filename + '!')

def save_split_dataset(filename):
	# splits data into 80-20 train-test set and saves as a pickle w/filename provided
	import pickle
	from sklearn.model_selection import train_test_split

	with open(filename, 'rb') as f:
		data = pickle.load(f)

	X = data['X']
	Y = data['Y']

	# one-hot encode Y | drop neutral
	temp = pd.get_dummies(Y).drop('neutral', axis=1)
	labels = temp.columns
	Y_split = temp.values

	# Split the final test set
	# 80-20 split | model-test
	X_model, X_test, Y_model, Y_test = train_test_split(
	    X, Y_split, random_state=7, stratify=Y_split, test_size=0.2)

	# Split the train and dev sets
	# 70-30 split | train-dev
	X_train, X_dev, Y_train, Y_dev = train_test_split(
	    X_model, Y_model, random_state=7, stratify=Y_model, test_size=0.3)

	split_data = {'X_train': X_train, 'X_dev': X_dev, 'X_test': X_test, 
	         'Y_train': Y_train, 'Y_dev': Y_dev, 'Y_test': Y_test, 
	         'labels': labels}

	if 'token' in filename:
		filename = './split_text_tokenized_data.pkl'
	else: 
		filename = './split_text_data.pkl'

	with open(filename, 'wb') as f:
		pickle.dump(split_data, f, pickle.HIGHEST_PROTOCOL)

	print('Saved ' + filename + '!')

def main():
	# Data Prep: cleaning
	df_tweets, df_isear = load_data_from_file()
	save_dataset(df_tweets, df_isear, './clean_text_data.pkl')
	save_dataset(df_tweets, df_isear, './clean_text_tokenized_data.pkl', if_featurize=True, if_tokenize=True)
	save_split_dataset('./clean_text_data.pkl')
	save_split_dataset('./clean_text_tokenized_data.pkl')

if __name__ == '__main__':
	main()
	