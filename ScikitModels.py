#Importing Numpy and Pandas
import numpy as np
import pandas as pd

#Importing Regular Expression & Argument Parser
import re
import argparse

#Importing NLTK Packages
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Importing Scikit-Learn Packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.pipeline

#Global Variables
train_input = 'data/train_input.csv'
train_output = 'data/train_output.csv'
test_input = 'data/test_input.csv'

class Process:
	def __init__(self,input_file = train_input, output_file = train_output, test_file = test_input):

		self.df = pd.read_csv(input_file)
		self.predict_df = pd.read_csv(test_file)
		self.output_df = pd.read_csv(output_file)

		self.Encode_Labels()
		self.Word_Vectorizer()

	def Encode_Labels(self):
		"""Encode labels with value between 0 and 7 (as we have 8 classes).
		Involves Fitting Label Encoder and then Transform it.
		Functions involved: Fit() and Transform()
		Reference: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html"""

		self.le = preprocessing.LabelEncoder()
		#Fit Label Encoder
		self.le.fit(['hockey','movies','nba','news','nfl','politics','soccer','worldnews'])
		#Return Encoded Labels
		self.y = self.le.transform(self.output_df.category)

	def Decode_Labels(self,y,filename='submission.csv'):
		"""This method involves decoding the labels and then write the output to a CSV file.
		Functions involved: Inverse_Transform()
		Reference: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html"""

		#Transform labels back to original encoding
		y_act = self.le.inverse_transform(y)

		self.predict_df['category'] = y_act
		#print self.predict_df['category']
		self.predict_df.drop(['conversation'], axis=1, inplace=True)
		#print self.predict_df['conversation']
		print 'No. of Rows Predicted = %d' % len(self.predict_df)
		self.predict_df.to_csv(filename, index=False)

	def Remove_Stopword(self, extract, remove_stopwords = False ):
		"""Removal of Stopwords from the list.
		Source (Stop words): http://www.ranks.nl/stopwords
		Source (Regular Expression): https://docs.python.org/2/library/re.html
		Reference: http://stackoverflow.com/questions/22763224/nltk-stopword-list"""

		extract = re.sub("[^a-zA-Z]"," ", extract)
		words = extract.lower().split()

		if remove_stopwords:
			#Reemoving the stop words from the dataset.
			stops = set(stopwords.words("english"))
			words = [w for w in words if not w in stops]

		return(words)

	def Word_Vectorizer(self):
		"""This function implements the WordNetLemmatizer.
		The words in training and test datasets are being lemmatized and then vectorized.
		Preprocessing.Normalize() is used to scale I/P individually to unit vector length"""

		lemmatized_words = WordNetLemmatizer()

		X_train, X_test, y_train, y_test = train_test_split (self.df.conversation, self.y, test_size = 0.33, random_state = 42)

		X_train_c = []
		X_test_c = []

		for s in X_train:
			X_train_c.append(" ".join(self.Remove_Stopword(s)))

		for s in X_test:
			X_test_c.append(" ".join(self.Remove_Stopword(s)))

		Pred_x = []
		for s in self.predict_df['conversation']:
			Pred_x.append(" ".join(self.Remove_Stopword(s)))
		stops = set(stopwords.words("english"))

		X_train_w = []
		for extract in X_train_c:
			words = extract.lower().split()
			words = [w for w in words if not w in stops]
			words = [lemmatized_words.lemmatize(w) for w in words]
			X_train_w.append(words)

		X_test_w = []
		for extract in X_test_c:
			words = extract.lower().split()
			words = [w for w in words if not w in stops]
			words = [lemmatized_words.lemmatize(w) for w in words]
			X_test_w.append(words)

		Pred_x_w = []
		for extract in Pred_x:
			words = extract.lower().split()
			words = [w for w in words if not w in stops]
			words = [lemmatized_words.lemmatize(w) for w in words]
			Pred_x_w.append(words)

		X_train_ws = []
		for extract in X_train_w:
			X_train_ws.append(" ".join(extract))

		X_test_ws = []
		for extract in X_test_w:
			X_test_ws.append(" ".join(extract))

		Pred_x_ws = []
		for extract in Pred_x_w:
			Pred_x_ws.append(" ".join(extract))

		print 'TF-IDF Vectorizer in progress...'

		tfidf_vectorizer = TfidfVectorizer( max_features = 40000, ngram_range = (1,3), sublinear_tf = True )

		X_train_tf = tfidf_vectorizer.fit_transform(X_train_ws)
		X_test_tf = tfidf_vectorizer.transform(X_test_ws)
		Pred_x_tf = tfidf_vectorizer.transform(Pred_x_ws)

		print 'Vectorization complete...'

		print "Normalization in progress..."

		self.X_train = preprocessing.normalize(X_train_tf, norm='l2')
		self.X_test = preprocessing.normalize(X_test_tf, norm='l2')
		self.Pred_x = preprocessing.normalize(Pred_x_tf, norm='l2')

		print "Normalization complete..."

		self.y_train = y_train
		self.y_test = y_test


class Runner():

	def __init__(self, args = None):

		self.prcs = Process(input_file = train_input, output_file = train_output, test_file = test_input)
		estimatorDict = {}

		if args.sgdc:
			#Support Vector Machines using Scikit Learn
			#Log - Logistic Regression
			#estimatorDict['SGDClassifier'] = SGDClassifier(loss = 'log', penalty = 'l2', n_jobs = -1, learning_rate = 'optimal', n_iter = 2000, alpha = 0.00001)
			#Hinge - Support Vector Machines
			#estimatorDict['SGDClassifier'] = SGDClassifier(loss = 'hinge', penalty = 'l2', n_jobs = -1, learning_rate = 'optimal', n_iter = 2000, alpha = 0.00001)
			#Modified_Huber - Smooth
			estimatorDict['SGDClassifier'] = SGDClassifier(loss = 'modified_huber', penalty = 'elasticnet', n_jobs = -1,learning_rate = 'optimal', n_iter = 2000, alpha = 0.00001)

		if args.rand:
			#Random Forest using Scikit Learn
			estimatorDict['RandomForest'] = RandomForestClassifier(n_jobs = -1, n_estimators = 1000, class_weight = 'balanced', max_features = 'log2')

		self.runClassifications(estimatorDict)

	def runClassifications(self,estimatorDict):

		for estimatorName in estimatorDict:

			print estimatorName + "in Progress..."
			print "Selecting K - Best features"
			select = sklearn.feature_selection.SelectKBest(k = 40000)

			clf = estimatorDict[estimatorName]

			steps = [('feature_selection', select),('random_forest', clf)]

			#Implementing Pipeline for Report Generation
			pipeline = sklearn.pipeline.Pipeline(steps)
			pipeline.fit(self.prcs.X_train,self.prcs.y_train)
			Pred_y = clf.predict(self.prcs.X_test)

			Accuracy = np.mean(Pred_y == self.prcs.y_test)

			report = sklearn.metrics.classification_report(self.prcs.y_test, Pred_y)
			#Preparing the report using pipeline
			print(report)

			print "Model's Accuracy: %f" % Accuracy

			print 'Prediction in progress...'

			y_fp = clf.predict(self.prcs.Pred_x)
			#Decoding the labels
			self.prcs.Decode_Labels(y_fp, estimatorName + '_PredictedOut' + '.csv')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-sgd','--sgdc', help = 'Stochastic Gradient Descent Classifier', action = 'store_true')
	parser.add_argument('-rf','--rand', help = 'Random Forest', action = 'store_true')

	args = parser.parse_args()
	Runner(args = args)
