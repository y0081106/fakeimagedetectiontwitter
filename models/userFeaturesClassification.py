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
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import time
import pprint

np.random.seed(0)

start_time = time.time()
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

def pipelineUserFeatures(user_df):
    userModelScores = []
    user_X_train = user_df.values[:,0:22]
    user_Y_train = user_df.values[:,22]
    num_trees = 100
    rfc = RandomForestClassifier(n_estimators=num_trees)
    user_model = rfc.fit(user_X_train, user_Y_train)
    return user_model


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

def read_merged_df(file_name):
    merged_df = pd.read_csv(file_name)
    user_new = merged_df.loc[merged_df['Indegree'] != 0]
    merged_df = UserLinearReg(merged_df, user_new)
    merged_df = normalizeUserFeatures(merged_df)
    #merged_df = merged_df.drop('id',1)
    cols = list(merged_df)
    cols.insert(23, cols.pop(cols.index('class')))
    merged_df = merged_df.ix[:, cols]
    event_data = merged_df.groupby('event')
    event_df = [event_data.get_group(x) for x in event_data.groups]
    return event_df

def prepare_data(event_df):
    for i,val in enumerate(event_df):
        test_data = pd.DataFrame(event_df[i])
        user_testing_data.append(test_data)
        event_train = event_df[:i]+event_df[i+1:]
        event_train = pd.concat(event_train)
        user_training_data.append(event_train)
    for i in range(0, len(user_training_data)):
        #user_training_data[i]= user_training_data[i].drop('tweet_id',1)
        user_training_data[i]= user_training_data[i].drop('event',1)
        #user_testing_data[i]= user_testing_data[i].drop('tweet_id',1)
        user_testing_data[i]= user_testing_data[i].drop('event',1)


    for i in range(0, len(user_training_data)):
        #print len(item_training_data[i])
        a = Counter(user_training_data[i]['class'])
        user_fake_count.append(a['fake']/9)
        user_real_count.append(a['real']/9)
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

def train_test_data(user_split):
    for i in range(0,len(user_split)):
        model_column = []
        for j in range(0,9):
            model = pipelineUserFeatures(user_split[i][j])
            model_column.append(model)
        user_Ms.append(model_column)

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
        
def calc_accuracy(user_fin_val):
    for i in range(0,len(user_fin_val)):
        user_acc_scores.append(accuracy_score(user_Y_test[i], user_fin_val[i]))
    return user_acc_scores
    
def calc_fake_accuracy(final_predictions):
    accuracy_val_fake = []
    cmat_total = []
    cmat_val = []
    from sklearn.metrics import confusion_matrix
    for i in range(len(final_predictions)):
        y_true = user_testing_data[i]['class']
        y_pred = final_predictions[i]
        cmat = confusion_matrix(y_true,y_pred)
        print cmat
        cmat_val.append(cmat[0][0])
        if len(cmat[0]>1):
            cmat_sum = 0
            for i in range(len(cmat)):
                cmat_sum = cmat_sum + cmat[0][i]
            cmat_total.append(cmat_sum)
        else:
            cmat_total.append(cmat[0])   
    #print cmat_val
    #print cmat_total
    for i in range(len(cmat_val)):
        accr = float(cmat_val[i])/cmat_total[i]
        accuracy_val_fake.append(float(accr))
    return accuracy_val_fake
    #avg = sum(accuracy_val_fake)/len(accuracy_val_fake)
    #print avg

event_df = read_merged_df("user_merged_with_ids_1.csv")
user_split = prepare_data(event_df)
final_predictions = train_test_data(user_split) 
return final_predictions
#acc_score = calc_accuracy(final_predictions)
#acc_fake_score = calc_fake_accuracy(final_predictions)
#print acc_fake_score
