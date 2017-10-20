import numpy as np
import pandas as pd
import time
import pprint

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

item_prediction_values = []
Ms = []
cv_scores = []
item_score = []
prediction_scores = []
item_combined_val = []
fake_count = []
real_count = []
item_fake = []
item_real = []
item_fake_split = []
item_real_split = []
item_split = []
item_training_data = []
item_testing_data = []
item_X_test = []
item_Y_test = []
item_prediction_values = []
testing_val = []
acc_scores = []

#Function that splits a DataFrame to a list of DataFrames of size n
def split_dataframe(df, n):
    n = int(n)
    df_size = len(df)
    batches = range(0, (df_size/n + 1) * n, n)
    return [df.iloc[i:i+n] for i in batches if i!=df_size] 

#Function that returns the model fitted on the training data
def pipeline(item_df):
    modelScores = []
    item_X_train = item_df.values[:,0:31]
    item_Y_train = item_df.values[:,31]
    num_trees = 100
    rfc = RandomForestClassifier(n_estimators=num_trees)
    item_model = rfc.fit(item_X_train, item_Y_train)
    return item_model

#Function that calculates linear regression
def LinearReg(item, item_new):
    
    item_new_1 = item_new[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase',
                       'numMentions', 'numHashtags' , 'numUrls' , 'positiveWords' , 'negativeWords',
                       'slangWords', 'rtCount', 'WotValue','numberNouns',  'readabilityValue']]
    item_1 = item[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase',
                       'numMentions', 'numHashtags' , 'numUrls' , 'positiveWords' , 'negativeWords',
                       'slangWords', 'rtCount', 'WotValue','numberNouns',  'readabilityValue']]
    lr = LinearRegression()
    X_train = item_new_1
    X_new_vals = item_1
    column_names = ['AlexaCountry','AlexaReach', 'AlexaDelta', 'AlexaPopularity', 'Harmonic', 'Indegree']
    Y_train = [item_new['AlexaCountry'],item_new['AlexaReach'],item_new['AlexaDelta'],item_new['AlexaPopularity'], item_new['Harmonic'], item_new['Indegree']]
    for i in range(0, len(Y_train)):
        linearmodel = lr.fit(X_train, Y_train[i]) 
        item[column_names[i]] = lr.predict(X_new_vals)
    return item
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
		
#Function that normalizes the feature range between -1 and 1
def normalize(item_df):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	item_df[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
			  'slangWords','rtCount','Indegree','Harmonic', 'AlexaPopularity', 'AlexaReach', 'AlexaDelta',
			  'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']] = scaler.fit_transform(item_df[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
			  'slangWords','rtCount','Indegree','Harmonic','AlexaPopularity', 'AlexaReach', 'AlexaDelta',
			  'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']])
	item_df = MultiColumnLabelEncoder(columns = ['questionSymbol','exclamSymbol','externLinkPresent','happyEmo', 'sadEmo', 'containFirstPron',
											'containSecPron', 'containThirdPron','colonSymbol','pleasePresent' ]).fit_transform(item_df)
	return item_df

#function that reads the csv file with the features and fills missing values and normalizes the values and places
#the class column at the end. It groups the columns by their event name and prepares separate dataframe for each event
def read_merged_df(file_name):
    merged_df = pd.read_csv(file_name)
    item_new = merged_df.loc[merged_df['AlexaCountry'] != 0]
    merged_df = LinearReg(merged_df, item_new)
    merged_df = normalize(merged_df)
    #merged_df = merged_df.drop('id',1)
    cols = list(merged_df)
    cols.insert(33, cols.pop(cols.index('class')))
    merged_df = merged_df.ix[:, cols]
    event_data = merged_df.groupby('event')
    event_df = [event_data.get_group(x) for x in event_data.groups]
    return event_df

#function to prepare training and testing data. Data belonging to one event is used as training and all others
#for testing in each iteration and is stored in item_testing_data and item_training_data. This is split into
#equivalent number of fake and real sets for the bagging technique.
def prepare_data(event_df):
    for i,val in enumerate(event_df):
        test_data = pd.DataFrame(event_df[i])
        item_testing_data.append(test_data)
        event_train = event_df[:i]+event_df[i+1:]
        event_train = pd.concat(event_train)
        item_training_data.append(event_train)


    for i in range(0, len(item_training_data)):
        #print len(item_training_data[i])
        a = Counter(item_training_data[i]['class'])
        fake_count.append(a['fake']/9)
        real_count.append(a['real']/9)
        item_df = item_training_data[i]
        item_fake.append(item_df.ix[item_df['class']=='fake'])
        item_real.append(item_df.ix[item_df['class']=='real'])

    for i in range(0, len(item_fake)):
        item_fake_split.append(split_dataframe(item_fake[i], (fake_count[i])+1))
        item_real_split.append(split_dataframe(item_real[i],(real_count[i])+1))
    
    for i in range(0, len(item_fake_split)):
        column = []
        for j in range(0, len(item_fake_split[i])):
            column.append(pd.concat([item_fake_split[i][j],item_real_split[i][j]]))
        item_split.append(column)
    return(item_split)

#function to train and test the split data. Calls the pipeline function to perform the supervised learning task. #Each learned model is stored in Ms and is used for prediction and majority voting takes place to determine the #final prediction.
def train_test_data(item_split):
    for i in range(0,len(item_split)):
        model_column = []
        for j in range(0,9):
            model = pipeline(item_split[i][j])
            model_column.append(model)
        Ms.append(model_column)

    for i in range(0, len(item_testing_data)):
        #item_testing_data[i] = item_testing_data[i].drop('event',1)
        item_df = item_testing_data[i]
        item_X_test.append(item_df.values[:,0:31])
        item_Y_test.append(item_df.values[:,31])

    item_prediction_values = []
    testing_val = []
    for i in range(0,len(Ms)):
        pred_column = []
        for j in range(0,9):
            pred_column.append(Ms[i][j].predict(item_X_test[i]))
        item_prediction_values.append(pred_column)
        testing_val.append(item_Y_test[i])

    pred_vals = []
    for i in range(len(Ms)):
        column1 = []
        for j in range(len(item_prediction_values[i][0])):
            column2 = []
            for k in range(0,9):
                column2.append(item_prediction_values[i][k][j])
            column1.append(column2)
        pred_vals.append(column1)

    fin_val = []
    res_key_value = []
    for i in range(0, len(pred_vals)):
        col = []
        for j in range(0,len(pred_vals[i])):
            result = Counter(pred_vals[i][j])
            res_key_val = result.keys(), result.values()
            col.append(res_key_val)
        res_key_value.append(col)

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
        fin_val.append(column)
    return fin_val

#Function that calculates the accuracy of both the true and fake values
def calc_accuracy(fin_val):
    for i in range(0,len(fin_val)):
        acc_scores.append(accuracy_score(item_Y_test[i], fin_val[i]))
    return acc_scores

#Function that calcuates the accuracy of only the fake values
def calc_fake_accuracy(final_predictions):
    accuracy_val_fake = []
    cmat_total = []
    cmat_val = []

    for i in range(len(final_predictions)):
        y_true = item_testing_data[i]['class']
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
    #print cmat_val
    #print cmat_total
    for i in range(len(cmat_val)):
        accr = float(cmat_val[i])/cmat_total[i]
        accuracy_val_fake.append(float(accr))
    return accuracy_val_fake



def read_args():
    if len(sys.argv) == 2:
        tweets_features = sys.argv[1]
    else:
        tweets_features = '../dataset/tweet_features_with_events.csv'
    return tweets_features

def main():
    #getting the file with the extracted tweet features
    fin = read_args()
    #divide into dataframes based on their events. Also, fill missing values and normalize.
    event_df = read_merged_df(fin)
    #prepare the data into training and testing data
    item_split = prepare_data(event_df)
    #take the training and testing data and run classification by calling pipeline. We also perform the majority vote
    #classification here
    final_predictions = train_test_data(item_split)
    print final_predictions
if __name__ == '__main_':
    main()