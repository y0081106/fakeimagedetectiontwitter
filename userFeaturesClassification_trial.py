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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import time
import pprint

def UserLinearReg(user_df, user_new):
    user_new_1 = user_new[['numFriends','numFollowers','followerFriendRatio','timesListed','numTweets','wotTrustValue','mediaContent',
                           'accountAge','tweetRatio']]
        
    user_df_1 = user_df[['numFriends','numFollowers','followerFriendRatio','timesListed','numTweets','wotTrustValue','mediaContent',
                           'accountAge','tweetRatio']]
    
    user_lr = LinearRegression()
    user_X_train = user_new_1
    user_new_vals = user_df_1
    user_column_names = ['alexaCountryRank','alexaDeltaRank','alexaPopularity','alexaReachRank', 'Harmonic', 'Indegree']
    user_Y_train = [user_new['alexaCountryRank'],user_new['alexaReachRank'],user_new['alexaDeltaRank'],user_new['alexaPopularity'],
         user_new['Harmonic'], user_new['Indegree']]
    for i in range(0, len(user_Y_train)):
        linearmodel = user_lr.fit(user_X_train, user_Y_train[i]) 
        user_df[user_column_names[i]] = user_lr.predict(user_new_vals)
    return user_df
	
def split_dataframe(df, n):
    """
    Helper function that splits a DataFrame to a list of DataFrames of size n

    :param df: pd.DataFrame
    :param n: int
    :return: list of pd.DataFrame
    """
    n = int(n)
    df_size = len(df)
    batches = range(0, (df_size/n + 1) * n, n)
    return [df.iloc[i:i+n] for i in batches if i!=df_size] 

user_url = "C:/Users/imaad/twitteradvancedsearch/user_features_all_1.txt"
user = pd.read_csv(user_url, sep=",", header = None, engine='python')
user.columns = ['numFriends', 'numFollowers', 'followerFriendRatio', 'timesListed', 'hasUrlCheck', 'verifiedUser','numTweets', 'bioCheck',
                'locationCheck', 'existingLocationCheck', 'wotTrustValue', 'mediaContent', 'accountAge', 'profileImgCheck', 'headerImgCheck', 
                'tweetRatio', 'Indegree', 'Harmonic','alexaCountryRank', 'alexaDeltaRank', 'alexaPopularity', 'alexaReachRank','class']
user = user.replace("?", "0")
user_new = user.loc[user['Indegree'] != 0]
user = UserLinearReg(user, user_new)
#user = normalizeUserFeatures(user)
user_fake = user.ix[user['class'] == 'fake']
user_real = user.ix[user['class'] == 'real']
#user_fake = shuffle(user_fake, random_state=100)
#user_real = shuffle(user_real, random_state=100)
#user_real.array_split(user_real,3)
user_fake_split = split_dataframe(user_fake, 1065)
user_real_split = split_dataframe(user_real, 1065)

if len(user_real_split) < 9:
    for i in range(0, (len(user_real_split))/2) :
        user_real_split.append(pd.concat([user_real_split[i],user_real_split[i+1]], ignore_index=True))
for i in range(0, len(user_real_split)):
    if len(user_real_split[i]) > 1065:
        user_real_split[i] = shuffle(user_real_split[i], random_state=60)
        user_real_split[i] = user_real_split[i].iloc[1065:]
    #print len(user_real_split[i])
#user_fake_split[0]
user_split = []
for i in range(0, len(user_fake_split)):
    #print len(user_fake_split[i])
    user_split.append(pd.concat([user_fake_split[8-i],user_real_split[i]]))
    #print len(user_split[i])

start_time = time.time()
item_prediction_values = []
userMs = []
item_score = []
prediction_scores = []

#Function that takes item features and normalizes the integer/float values, encodes the string values
def normalizeUserFeatures(user_df):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	
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
	rfc = RandomForestClassifier(n_estimators=num_trees)
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

		
for i in range(0,9):
    user_split[i] = normalizeUserFeatures(user_split[i])
    user_model = pipelineUserFeatures(user_split[i])
    userMs.append(user_model)
#print userMs
def userClassification():
	user_url_test = "C:/Users/imaad/twitteradvancedsearch/user_features_all_test_1.txt"
	user_all_test = pd.read_csv(user_url_test, sep=",", header = None, engine='python')
	user_all_test.columns = ['numFriends', 'numFollowers', 'followerFriendRatio', 'timesListed', 'hasUrlCheck', 'verifiedUser','numTweets', 'bioCheck',
					'locationCheck', 'existingLocationCheck', 'wotTrustValue', 'mediaContent', 'accountAge', 'profileImgCheck', 'headerImgCheck', 
					'tweetRatio', 'Indegree', 'Harmonic','alexaCountryRank', 'alexaDeltaRank', 'alexaPopularity', 'alexaReachRank','class']
	user_all_test = user_all_test.replace("?", 0)
	#user_all_test = user_all_test.fillna(user_all_test.mean())
	#user_all_test = UserLinearReg(user_all_test)
	user_all_test = UserLinearReg(user_all_test, user_new)
	user_all_test = normalizeUserFeatures(user_all_test)
	user_all_X_test = user_all_test.values[:,0:22]
	user_all_Y_test = user_all_test.values[:,22]



	predscore_item = []
	prediction_values1, user_prediction_values, user_testing_val = [],[], []
	user_score = []
	for m in userMs:    
		user_all_predval = m.predict(user_all_X_test)
		user_predscore = m.predict_proba(user_all_X_test)
		user_testing_val.append(user_all_Y_test)
		user_prediction_values.append(user_all_predval)
		user_score.append(user_predscore)

	#print user_prediction_values
	#print user_score
	prediction_val = []
	user_combined_val = []
	abc1 = []
	for i in range(0,len(userMs)):
			column = []
			for j in range(0,len(user_all_Y_test)):
					if (user_score[i][j][0]) > (user_score[i][j][1]):
						column.append(user_score[i][j][0])
					else:
						column.append(user_score[i][j][1])
			prediction_val.append(column)

	for i in range(0,len(userMs)):
			column1 = []
			for j in range(0,len(user_all_Y_test)):
				abc1 = user_prediction_values[i][j], user_testing_val[i][j], user_prediction_values[i][j]
				column1.append(abc1)
			user_combined_val.append(column1)
	   
	#print (user_combined_val)
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

print "Time taken:", time.time() - start_time