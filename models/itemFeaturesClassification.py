import numpy as np
import pandas as pd
import sys

from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)

NUM_MODELS = 9

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
	#select only the first 32 values for X and the last one for Y
    item_X_train = item_df.values[:,0:31]
    item_Y_train = item_df.values[:,31]
    num_trees = 100
    rfc = RandomForestClassifier(n_estimators=num_trees)
    item_model = rfc.fit(item_X_train, item_Y_train)
    return item_model

#Function that calculates linear regression
def LinearReg(item):
    item_new = item.loc[item['AlexaCountry'] != 0]
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

#function that fills missing values and normalizes the values and places
#the class column at the end. It groups the columns by their event name and prepares separate dataframe for each event	
def preprocess(item_df):
    item_df = LinearReg(item_df)
    item_df = normalize(item_df)
    cols = list(item_df)
    #inserting class at the end of the dataframe
    cols.insert(len(cols), cols.pop(cols.index('class')))
    item_df = item_df.ix[:, cols]
    event_data = item_df.groupby('event')
    event_df = [event_data.get_group(x) for x in event_data.groups]
    for i in range(len(event_df)):
        event_df[i] = event_df[i].drop('event',1)
    return event_df

#function to prepare training and testing data. Data belonging to one event is used as training and all others
#for testing in each iteration and is stored in item_testing_data and item_training_data. This is split into
#equivalent number of fake and real sets for the bagging technique.
def prepare(event_df):
    for i,val in enumerate(event_df):
        test_data = pd.DataFrame(event_df[i])
		#item_testing_data is a list with data from each event. To be used for testing
        item_testing_data.append(test_data)
		#selecting training data for all events except for the one being tested
        event_train = event_df[:i]+event_df[i+1:]
		#concatenating all event dataframes into one
        event_train = pd.concat(event_train)
		#item_training_data is a list with data from all events except the one being tested
        item_training_data.append(event_train)

	#iterate over item_training_data. item_training_data (in our case) has a len of 17 as it has all training samples in each 
	#iteration except the one being tested
    for i in range(len(item_training_data)):
		#Count the number of classes in each iteration
        a = Counter(item_training_data[i]['class'])
		#fake_count and real_count has the number of fake and real classes in each iteration for each model
        fake_count.append(a['fake']/NUM_MODELS)
        real_count.append(a['real']/NUM_MODELS)
		#item_fake and item_real has the fake and real classes in each iteration
        item_fake.append(item_training_data[i].ix[item_training_data[i]['class']=='fake'])
        item_real.append(item_training_data[i].ix[item_training_data[i]['class']=='real'])

    for i in range(len(item_fake)):
		#item_fake_split and item_real_split has the split dataframe for each iteration and for each model
        item_fake_split.append(split_dataframe(item_fake[i], (fake_count[i])+1))
        item_real_split.append(split_dataframe(item_real[i],(real_count[i])+1))
    
	#2D List containing 9 instances of concatenated fake and real tweets for each event (17 in total)
    for i in range(len(item_fake_split)):
        column = []
        for j in range(0, len(item_fake_split[i])):
            column.append(pd.concat([item_fake_split[i][j],item_real_split[i][j]]))
        item_split.append(column)
    return(item_split)

#functions to train and test the split data. Calls the pipeline function to perform the supervised learning task. #Each learned model is stored in Ms and is used for prediction and majority voting takes place to determine the #final prediction.

def fit(item_split):
    for i in range(0,len(item_split)):
        model_column = []
        for j in range(0,NUM_MODELS):
            model = pipeline(item_split[i][j])
            model_column.append(model)
        Ms.append(model_column)
    models = Ms 
    return models
	
def predict(Ms):
    for i in range(0, len(item_testing_data)):
        item_df = item_testing_data[i]
        item_X_test.append(item_df.values[:,0:31])
        item_Y_test.append(item_df.values[:,31])

    item_prediction_values = []
    testing_val = []
    for i in range(0,len(Ms)):
        pred_column = []
        for j in range(0,NUM_MODELS):
            pred_column.append(Ms[i][j].predict(item_X_test[i]))
		#item_prediction_values contains the predictions. i loop contains the classifier number, j loop the predictions
        item_prediction_values.append(pred_column)
		#adding the true values to testing_val
        testing_val.append(item_Y_test[i])

    pred_vals = []
    for i in range(len(Ms)):
        column1 = []
        for j in range(len(item_prediction_values[i][0])):
            column2 = []
            for k in range(0,NUM_MODELS):
                column2.append(item_prediction_values[i][k][j])
            column1.append(column2)
        pred_vals.append(column1)

    fin_val = []
    res_key_value = []
	#2D list with predictions from each of the classifier, i loop contains the classifiers, j loops the samples that were #predicted in each case
    for i in range(0, len(pred_vals)):
        col = []
        for j in range(0,len(pred_vals[i])):
            result = Counter(pred_vals[i][j])
            res_key_val = result.keys(), result.values()
            col.append(res_key_val)
        res_key_value.append(col)
		
	#majority vote prediction
    for i in range(0,len(pred_vals)):
        column = []
        for j in range(0, len(pred_vals[i])):
		# if both fake and real have been predicted then len is >= 2 
            if len(res_key_value[i][j][0]) >= 2:
			#If count for real is greater than count for fake
                if res_key_value[i][j][1][0] > res_key_value[i][j][1][1]:
                    column.append("real")
                else:
                    column.append("fake")
            else:
			#else append the first value in res_key_value since there was only one prediction by all the 9 instances
                column.append(res_key_value[i][j][0][0])
        fin_val.append(column)
    return fin_val
	
#Function that calcuates the accuracy of only the fake values
def accuracy(final_predictions):
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
	
def final_pred():
	file_name = read_args()
	df = pd.read_csv(file_name)
	df = preprocess(df)
	df = prepare(df)
	models = fit(df)
	final_predictions = predict(models)
	return final_predictions

def main():
	#getting the file with the extracted tweet features
	print "get file name"
	file_name = read_args()
	#read in csv with tweet features
	print "read the file"
	df = pd.read_csv(file_name)
	#preprocess it (Linear regression for missing values and normalizing the numeric values)
	print "pre processing"
	df = preprocess(df)
	#prepare data for training and testing (split data samples based on class)
	print "preparing data"
	df = prepare(df)
	#train 
	print "training"
	models = fit(df)
	#predict
	print "prediction"
	final_predictions = predict(models)
	#calculate accuracy
	print "calc_accuracy"
	acc = accuracy(final_predictions)
	#return final_predictions
	avg = sum(acc)/len(acc)
	print avg
	
if __name__ == '__main__':
    main()