import numpy as np
import pandas as pd
import time
import pprint
import sys

from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)

NUM_MODELS = 9

user_prediction_values = []
user_Ms = []
user_cv_scores = []
user_score = []
user_prediction_scores = []
user_combined_val = []
user_fake_count = []
user_real_count = []
user_fake = []
user_real = []
user_fake_split = []
user_real_split = []
user_split = []
user_training_data = []
user_testing_data = []
user_X_test = []
user_Y_test = []
user_prediction_values = []
user_testing_val = []
user_acc_scores = []

#Function that splits a DataFrame to a list of DataFrames of size n
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

#Function that returns the model fitted on the training data
def pipelineUserFeatures(user_df):
	#select only the first 23 values for X and the last one for Y
    user_X_train = user_df.values[:,0:22]
    user_Y_train = user_df.values[:,22]
    num_trees = 100
    rfc = RandomForestClassifier(n_estimators=num_trees)
    user_model = rfc.fit(user_X_train, user_Y_train)
    return user_model

#Function that calculates linear regression
def UserLinearReg(user_df):
    user_new = user_df.loc[user_df['Indegree'] != 0]
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

#class that encodes columns with string values to numeric values
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

#Function that takes item features and normalizes the integer/float values, encodes the string values
def normalizeUserFeatures(user_df):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	
	user_df[['numFriends','numFollowers','followerFriendRatio','timesListed','numTweets','wotTrustValue','mediaContent','accountAge',
			 'tweetRatio','Indegree','Harmonic','alexaCountryRank','alexaDeltaRank','alexaPopularity','alexaReachRank']] = scaler.fit_transform(user_df[['numFriends','numFollowers','followerFriendRatio','timesListed','numTweets','wotTrustValue','mediaContent','accountAge',
			 'tweetRatio','Indegree','Harmonic','alexaCountryRank','alexaDeltaRank','alexaPopularity','alexaReachRank']])

	user_df = MultiColumnLabelEncoder(columns = ['hasUrlCheck','verifiedUser','bioCheck','locationCheck','existingLocationCheck',
												   'profileImgCheck','headerImgCheck']).fit_transform(user_df)
	return user_df
	
#function that fills missing values and normalizes the values and places
#the class column at the end. It groups the columns by their event name and prepares separate dataframe for each event	
def pre_processing(user_df):
    user_df = UserLinearReg(user_df)
    user_df = normalizeUserFeatures(user_df)
    cols = list(user_df)
    #inserting class at the end of the dataframe
    cols.insert(len(cols), cols.pop(cols.index('class')))
    user_df = user_df.ix[:, cols]
    event_data = user_df.groupby('event')
    event_df = [event_data.get_group(x) for x in event_data.groups]
    for i in range(len(event_df)):
        event_df[i] = event_df[i].drop('event',1)
    return event_df

#function to prepare training and testing data. Data belonging to one event is used as training and all others
#for testing in each iteration and is stored in user_testing_data and user_training_data. This is split into
#equivalent number of fake and real sets for the bagging technique.
def prepare_data(event_df):
    for i,val in enumerate(event_df):
        test_data = pd.DataFrame(event_df[i])
		#user_testing_data is a list with data from each event. To be used for testing
        user_testing_data.append(test_data)
		#selecting training data for all events except for the one being tested
        event_train = event_df[:i]+event_df[i+1:]
		#concatenating all event dataframes into one
        event_train = pd.concat(event_train)
		#user_training_data is a list with data from all events except the one being tested
        user_training_data.append(event_train)

	#iterate over item_training_data. user_training_data (in our case) has a len of 17 as it has all training samples in each 
	#iteration except the one being tested	
    for i in range(0, len(user_training_data)):
        a = Counter(user_training_data[i]['class'])
        user_fake_count.append(a['fake']/NUM_MODELS)
        user_real_count.append(a['real']/NUM_MODELS)
        user_df = user_training_data[i]
        user_fake.append(user_df.ix[user_df['class']=='fake'])
        user_real.append(user_df.ix[user_df['class']=='real'])

    for i in range(0, len(user_fake)):
        user_fake_split.append(split_dataframe(user_fake[i], (user_fake_count[i])+1))
        user_real_split.append(split_dataframe(user_real[i],(user_real_count[i])+1))

    for i in range(0, len(user_fake_split)):
        column = []
        for j in range(0, len(user_fake_split[i])):
            column.append(pd.concat([user_fake_split[i][j],user_real_split[i][j]]))
        user_split.append(column)
    return(user_split)

def train_data(user_split):
    for i in range(0,len(user_split)):
        model_column = []
        for j in range(0,NUM_MODELS):
            model = pipelineUserFeatures(user_split[i][j])
            model_column.append(model)
        user_Ms.append(model_column)
	models = user_Ms 
    return models

def test_data(user_Ms):
    for i in range(0, len(user_testing_data)):
        #item_testing_data[i] = item_testing_data[i].drop('event',1)
        user_df = user_testing_data[i]
        user_X_test.append(user_df.values[:,0:22])
        user_Y_test.append(user_df.values[:,22])

    user_prediction_values = []
    user_testing_val = []
    for i in range(0,len(user_Ms)):
        pred_column = []
        for j in range(0,9):
            pred_column.append(user_Ms[i][j].predict(user_X_test[i]))
        user_prediction_values.append(pred_column)
        user_testing_val.append(user_Y_test[i])

    pred_vals = []
    for i in range(0, len(user_Ms)):
        column1 = []
        for j in range(0,len(user_prediction_values[i][0])):
            column2 = []
            for k in range(0,9):
                column2.append(user_prediction_values[i][k][j])
            column1.append(column2)
        pred_vals.append(column1)

    user_fin_val = []
    res_key_value = []
    for i in range(0, len(pred_vals)):
        col = []
        for j in range(0,len(pred_vals[i])):
            result = Counter(pred_vals[i][j])
            res_key_val = result.keys(), result.values()
            col.append(res_key_val)
        res_key_value.append(col)
    #pprint.pprint((res_key_value[0]))
        #print res_key_value[1][1][0]

    for i in range(0,len(pred_vals)):
        column = []
        for j in range(0, len(pred_vals[i])):
        #print len(res_key_value[i])
            if len(res_key_value[i][j][0]) >= 2:
                if res_key_value[i][j][1][0] > res_key_value[i][j][1][1]:
                    column.append("real")
                else:
                    column.append("fake")
            else:
                column.append(res_key_value[i][j][0][0])
        user_fin_val.append(column)
    return user_fin_val
        
def calc_accuracy(final_predictions):
    accuracy_val_fake = []
    cmat_total = []
    cmat_val = []
    from sklearn.metrics import confusion_matrix
    for i in range(len(final_predictions)):
        y_true = user_testing_data[i]['class']
        y_pred = final_predictions[i]
        cmat = confusion_matrix(y_true,y_pred)
        cmat_val.append(cmat[0][0])
        if len(cmat[0]>1):
            cmat_sum = 0
            for i in range(len(cmat)):
                cmat_sum = cmat_sum + cmat[0][i]
            cmat_total.append(cmat_sum)
        else:
            cmat_total.append(cmat[0])   
    for i in range(len(cmat_val)):
        accr = float(cmat_val[i])/cmat_total[i]
        accuracy_val_fake.append(float(accr))
    return accuracy_val_fake

def read_args():
    if len(sys.argv) == 2:
        user_features = sys.argv[1]
    else:
        user_features ='C:/Users/imaad/twitteradvancedsearch/fakeimagedetectiontwitter/dataset/user_features_with_events.csv'
    return user_features

def main():
	#getting the file with the extracted tweet features
	print "get file name"
	file_name = read_args()
	#read in csv with tweet features
	print "read the file"
	df = pd.read_csv(file_name)
	#preprocess it (Linear regression for missing values and normalizing the numeric values)
	print "pre processing"
	df = pre_processing(df)
	#prepare data for training and testing
	print "preparing data"
	df = prepare_data(df)
	#train 
	print "training"
	models = train_data(df)
	#predict
	print "prediction"
	final_predictions = test_data(models)
	#calculate accuracy
	print "calc_accuracy"
	acc = calc_accuracy(final_predictions)
	#return final_predictions
	print acc
	
if __name__ == '__main__':
    main()