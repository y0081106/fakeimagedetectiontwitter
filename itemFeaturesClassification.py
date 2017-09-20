from collections import Counter
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer
#from sklearn import datasets, linear_model, cross_validation, grid_search
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import pprint

start_time = time.time()

def itemClassification():

	item_prediction_values = []
	Ms = []
	cv_scores = []
	item_score = []
	prediction_scores = []
	#Function that takes item features and normalizes the integer/float values, encodes the string values
	def normalize(item_df):
		scaler = MinMaxScaler()
		item_df[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
				  'slangWords','rtCount', 'AlexaPopularity', 'AlexaReach', 'AlexaDelta',
				  'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']] = scaler.fit_transform(item_df[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
				  'slangWords','rtCount', 'AlexaPopularity', 'AlexaReach', 'AlexaDelta',
				  'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']])
		item_df = MultiColumnLabelEncoder(columns = ['questionSymbol','exclamSymbol','externLinkPresent','happyEmo', 'sadEmo', 'containFirstPron',
												'containSecPron', 'containThirdPron','colonSymbol','pleasePresent' ]).fit_transform(item_df)
		return item_df

	def LinearReg(item_df):
		item_df_new = item_df[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase',
							  'numMentions', 'numHashtags' , 'numUrls' , 'positiveWords' , 'negativeWords',
							  'slangWords', 'rtCount', 'WotValue','numberNouns',  'readabilityValue']]
		
		lr = LinearRegression()
		X = item_df_new
		column_names = ['AlexaCountry','AlexaReach', 'AlexaDelta', 'AlexaPopularity', 'Harmonic', 'Indegree']
		Y = [item_df['AlexaCountry'],item_df['AlexaReach'],item_df['AlexaDelta'],item_df['AlexaPopularity'], item_df['Harmonic'], item_df['Indegree']]
		for i in range(0, len(Y)):
			linearmodel = lr.fit(X, Y[i]) 
			item_df_new[column_names[i]] = lr.predict(X)
		for i in range(0,len(column_names)):
			item_df[column_names[i]] = item_df_new[column_names[i]]
		return item_df



	def pipeline(item_df):
		#modelScores = []
		item_X_train = item_df.values[:,0:31]
		item_Y_train = item_df.values[:,31]
		num_trees = 100
		
		rfc = RandomForestClassifier(n_estimators=num_trees, max_features= 1,random_state = 84, min_samples_split = 40,max_leaf_nodes = 15, max_depth = 5,min_samples_leaf = 3)
		item_model = rfc.fit(item_X_train, item_Y_train)
		#scores = cross_val_score(item_model, item_X_train, item_Y_train, cv = 2)
		#modelScores.append(item_model)
		#modelScores.append(scores)
		return item_model

	#class to perform multicolumn label encoding   
	class MultiColumnLabelEncoder:
		def __init__(self,columns = None):
			self.columns = columns # array of column names to encode

		def fit(self,X,y=None):
			return self # not relevant here

		def transform(self,X):
			'''
			Transforms columns of X specified in self.columns using
			LabelEncoder(). If no columns specified, transforms all
			columns in X.
			'''
			output = X.copy()
			if self.columns is not None:
				for col in self.columns:
					output[col] = LabelEncoder().fit_transform(output[col])
			else:
				for colname,col in output.iteritems():
					output[colname] = LabelEncoder().fit_transform(col)
			return output

		def fit_transform(self,X,y=None):
			return self.fit(X,y).transform(X)

		

	item_url = "C:/Users/imaad/twitteradvancedsearch/item_features_all_1.txt"
	item = pd.read_csv(item_url, sep=",", header = None, engine='python')
	item.columns = ['tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol', 'externLinkPresent','numberNouns', 'happyEmo',
			  'sadEmo', 'containFirstPron','containSecPron','containThirdPron','numUpperCase', 'positiveWords', 'negativeWords', 
					   'numMentions', 'numHashtags', 'numUrls','rtCount' ,
					'slangWords','colonSymbol','pleasePresent' ,'WotValue', 'numQuesSymbol','numExclamSymbol', 'readabilityValue','Indegree',
					   'Harmonic' ,'AlexaCountry','AlexaDelta','AlexaPopularity', 'AlexaReach' , 'class']

	item = item.replace("?", "0")
	#item = LinearReg(item)
	item = normalize(item)

	item_url_test = "C:/Users/imaad/twitteradvancedsearch/item_features_all_test_1.txt"
	item_all_test = pd.read_csv(item_url_test, sep=",", header = None, engine='python')
	item_all_test.columns = ['tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol', 'externLinkPresent','numberNouns', 'happyEmo',
			  'sadEmo', 'containFirstPron','containSecPron','containThirdPron','numUpperCase', 'positiveWords', 'negativeWords', 
					   'numMentions', 'numHashtags', 'numUrls','rtCount' ,
					'slangWords','colonSymbol','pleasePresent' ,'WotValue', 'numQuesSymbol','numExclamSymbol', 'readabilityValue','Indegree',
					   'Harmonic' ,'AlexaCountry','AlexaDelta','AlexaPopularity', 'AlexaReach' , 'class']


	item_all_test = item_all_test.replace("?", "0")
	#item_all_test = LinearReg(item_all_test)
	item_all_test = normalize(item_all_test)
	item_all_X_test = item_all_test.values[:,0:31]
	item_all_Y_test = item_all_test.values[:,31]
	
	for i in range(0,9):
		
		item = shuffle(item, random_state=3000)
		model = pipeline(item)
		Ms.append(model)
		#cv_scores.append(model[1])
	#print Ms

	predscore_item = []
	prediction_values1, item_prediction_values, testing_val = [],[], []
	for m in Ms:
		item_all_predval = m.predict(item_all_X_test)
		testing_val.append(item_all_Y_test)
		item_prediction_values.append(item_all_predval)
		

	prediction_val = []
	item_combined_val = []
	abc = []

	for i in range(0,len(Ms)):
		item_column = []
		for j in range(0,len(item_all_X_test)):
			abc = item_prediction_values[i][j], testing_val[i][j]
			item_column.append(abc)
		item_combined_val.append(item_column)
				
	#print(item_combined_val)


	cd = []
	a = []
	for i in range(0, len(item_all_Y_test)):
		ab =[]
		for j in range(0, len(Ms)):
			ab.append(item_combined_val[j][i][0])
		cd.append(ab)
	for i in range(0, len(item_all_Y_test)):
		res = Counter(cd[i])
		b = res.keys(), res.values()
		a.append(b),
		#a.append(res.values())
		#print Counter(res).keys() 
		#print Counter(res).values() 
	#print len(a[0][0])
	finVal = []
	for i in range(0 ,len(a)):
		if len(a[i][0])>=2:
			if a[i][1][0] > a[i][1][1]:
				finVal.append("real")
			else:
				finVal.append("fake")
		else:
			finVal.append(a[i][0][0])
	return (finVal)
#itemClassification()