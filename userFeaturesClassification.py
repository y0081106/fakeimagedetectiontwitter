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


def userClassification():
	start_time = time.time()
	item_prediction_values = []
	userMs = []
	item_score = []
	prediction_scores = []
	def UserLinearReg(user_df):
		user_df_new = user_df[['numFriends','numFollowers','followerFriendRatio','timesListed','numTweets','wotTrustValue','mediaContent',
							   'accountAge','tweetRatio']]
		
		user_lr = LinearRegression()
		user_X = user_df_new
		user_column_names = ['alexaCountryRank','alexaDeltaRank','alexaPopularity','alexaReachRank', 'Harmonic', 'Indegree']
		user_Y = [user_df['alexaCountryRank'],user_df['alexaReachRank'],user_df['alexaDeltaRank'],user_df['alexaPopularity'],
			 user_df['Harmonic'], user_df['Indegree']]
		for i in range(0, len(user_Y)):
			linearmodel = user_lr.fit(user_X, user_Y[i]) 
			user_df_new[user_column_names[i]] = user_lr.predict(user_X)
		for i in range(0,len(user_column_names)):
			user_df[user_column_names[i]] = user_df_new[user_column_names[i]]
		return user_df
		
	#Function that takes item features and normalizes the integer/float values, encodes the string values
	def normalizeUserFeatures(user_df):
		scaler = MinMaxScaler()
		
		user_df[['numFriends','numFollowers','followerFriendRatio','timesListed','numTweets','wotTrustValue','mediaContent','accountAge',
				 'tweetRatio','Indegree','Harmonic','alexaCountryRank','alexaDeltaRank','alexaPopularity','alexaReachRank']] = scaler.fit_transform(user_df[['numFriends','numFollowers','followerFriendRatio','timesListed','numTweets','wotTrustValue','mediaContent','accountAge',
				 'tweetRatio','Indegree','Harmonic','alexaCountryRank','alexaDeltaRank','alexaPopularity','alexaReachRank']])

		user_df = MultiColumnLabelEncoder(columns = ['hasUrlCheck','verifiedUser','bioCheck','locationCheck','existingLocationCheck',
													   'profileImgCheck','headerImgCheck']).fit_transform(user_df)
		return user_df

	#function that converts the training data into X and Y samples and builds the Random Forest model
	def pipelineUserFeatures(user_df):
		user_X_train = user_df.values[:,0:22]
		user_Y_train = user_df.values[:,22]
		num_trees = 100
		rfc = RandomForestClassifier(n_estimators=num_trees, max_features= 1,random_state = 84, min_samples_split = 40,max_leaf_nodes = 15, max_depth = 5,min_samples_leaf = 3)
		user_model = rfc.fit(user_X_train, user_Y_train)
		#user_scores = cross_val_score(user_model, user_X_train, user_Y_train, cv = 2)
		return user_model
		
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

		

	user_url = "C:/Users/imaad/twitteradvancedsearch/user_features_all_1.txt"
	user = pd.read_csv(user_url, sep=",", header = None, engine='python')
	user.columns = ['numFriends', 'numFollowers', 'followerFriendRatio', 'timesListed', 'hasUrlCheck', 'verifiedUser','numTweets', 'bioCheck',
					'locationCheck', 'existingLocationCheck', 'wotTrustValue', 'mediaContent', 'accountAge', 'profileImgCheck', 'headerImgCheck', 
					'tweetRatio', 'Indegree', 'Harmonic','alexaCountryRank', 'alexaDeltaRank', 'alexaPopularity', 'alexaReachRank','class']
	user = user.replace("?", "0")
	#user = UserLinearReg(user)
	user = normalizeUserFeatures(user)
	
	for i in range(0,9):
		user = shuffle(user, random_state=3000)
		user_model = pipelineUserFeatures(user)
		userMs.append(user_model)
	#print userMs

	user_url_test = "C:/Users/imaad/twitteradvancedsearch/user_features_all_test_1.txt"
	user_all_test = pd.read_csv(user_url_test, sep=",", header = None, engine='python')
	user_all_test.columns = ['numFriends', 'numFollowers', 'followerFriendRatio', 'timesListed', 'hasUrlCheck', 'verifiedUser','numTweets', 'bioCheck',
					'locationCheck', 'existingLocationCheck', 'wotTrustValue', 'mediaContent', 'accountAge', 'profileImgCheck', 'headerImgCheck', 
					'tweetRatio', 'Indegree', 'Harmonic','alexaCountryRank', 'alexaDeltaRank', 'alexaPopularity', 'alexaReachRank','class']
	user_all_test = user_all_test.replace("?", "0")
	#user_all_test = UserLinearReg(user_all_test)
	user_all_test = normalizeUserFeatures(user_all_test)
	user_all_X_test = user_all_test.values[:,0:22]
	user_all_Y_test = user_all_test.values[:,22]



	predscore_item = []
	prediction_values1, user_prediction_values, user_testing_val = [],[], []
	for m in userMs:
		user_all_predval = m.predict(user_all_X_test)
		user_testing_val.append(user_all_Y_test)
		user_prediction_values.append(user_all_predval)
		

	prediction_val = []
	user_combined_val = []
	cde = []

	for i in range(0,len(userMs)):
		user_column = []
		for j in range(0,len(user_all_X_test)):
			cde = user_prediction_values[i][j], user_testing_val[i][j]
			user_column.append(cde)
		user_combined_val.append(user_column)
				
	#print(user_combined_val)

	user_cd = []
	user_a = []
	for i in range(0, len(user_all_Y_test)):
		user_ab =[]
		for j in range(0, len(userMs)):
			user_ab.append(user_combined_val[j][i][0])
		user_cd.append(user_ab)
	for i in range(0, len(user_all_Y_test)):
		user_res = Counter(user_cd[i])
		user_b = user_res.keys(), user_res.values()
		user_a.append(user_b)
		#a.append(res.values())
		#print Counter(res).keys() 
		#print Counter(res).values() 
	#print len(a[0][0])
	user_finVal = []
	for i in range(0 ,len(user_a)):
		if len(user_a[i][0])>=2:
			if user_a[i][1][0] > user_a[i][1][1]:
				user_finVal.append("real")
			else:
				user_finVal.append("fake")
		else:
			user_finVal.append(user_a[i][0][0])
	return (user_finVal)