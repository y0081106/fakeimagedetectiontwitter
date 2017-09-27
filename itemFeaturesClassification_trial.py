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

item_url = "C:/Users/imaad/twitteradvancedsearch/item_features_all_1.txt"
item = pd.read_csv(item_url, sep=",", header = None, engine='python')
item.columns = ['tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol', 'externLinkPresent','numberNouns', 'happyEmo',
          'sadEmo', 'containFirstPron','containSecPron','containThirdPron','numUpperCase', 'positiveWords', 'negativeWords', 
                   'numMentions', 'numHashtags', 'numUrls','rtCount' ,
                'slangWords','colonSymbol','pleasePresent' ,'WotValue', 'numQuesSymbol','numExclamSymbol', 'readabilityValue','Indegree',
                   'Harmonic' ,'AlexaCountry','AlexaDelta','AlexaPopularity', 'AlexaReach' , 'class']

item = item.replace('?', 0)
item_new = item.loc[item['AlexaCountry'] != 0]
item = LinearReg(item, item_new)
item_fake = item.ix[item['class'] == 'fake']
item_real = item.ix[item['class'] == 'real']
#item_fake = shuffle(item_fake, random_state=70)
#item_real = shuffle(item_real, random_state=70)
#item_real.array_split(item_real,3)
item_fake_split = split_dataframe(item_fake, 1065)
item_real_split = split_dataframe(item_real, 1065)

if len(item_real_split) < 9:
    for i in range(0, (len(item_real_split))/2) :
        item_real_split.append(pd.concat([item_real_split[i],item_real_split[i+1]], ignore_index=True))
for i in range(0, len(item_real_split)):
    if len(item_real_split[i]) > 1065:
        item_real_split[i] = shuffle(item_real_split[i], random_state=100)
        item_real_split[i] = item_real_split[i].iloc[1065:]
    #print len(item_real_split[i])
#item_fake_split[0]
item_split = []
for i in range(0, len(item_fake_split)):
    #print len(item_fake_split[i])
    item_split.append(pd.concat([item_fake_split[i],item_real_split[8-i]]))
    #print len(item_split[i])

start_time = time.time()
item_prediction_values = []
Ms = []
cv_scores = []
item_score = []
prediction_scores = []
#Function that takes item features and normalizes the integer/float values, encodes the string values
def normalize(item_df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    item_df[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
              'slangWords','rtCount', 'Indegree','Harmonic','AlexaPopularity', 'AlexaReach', 'AlexaDelta',
              'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']] = scaler.fit_transform(item_df[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
              'slangWords','rtCount','Indegree','Harmonic', 'AlexaPopularity', 'AlexaReach', 'AlexaDelta',
              'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']])
    item_df = MultiColumnLabelEncoder(columns = ['questionSymbol','exclamSymbol','externLinkPresent','happyEmo', 'sadEmo', 'containFirstPron',
                                            'containSecPron', 'containThirdPron','colonSymbol','pleasePresent' ]).fit_transform(item_df)
    return item_df


def pipeline(item_df):
    #modelScores = []
    item_X_train = item_df.values[:,0:31]
    item_Y_train = item_df.values[:,31]
    num_trees = 100
    rfc = RandomForestClassifier(n_estimators=num_trees)
    item_model = rfc.fit(item_X_train, item_Y_train)
    #scores = cross_val_score(item_model, item_X_train, item_Y_train, cv = 3)
    #modelScores.append(item_model)
    #modelScores.append(scores.mean())
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

for i in range(0, len(item_split)):
    item_split[i] = normalize(item_split[i])
	
def itemClassification():
	item_url_test = "C:/Users/imaad/twitteradvancedsearch/item_features_all_test_1.txt"
	item_all_test = pd.read_csv(item_url_test, sep=",", header = None, engine='python')
	item_all_test.columns = ['tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol', 'externLinkPresent','numberNouns', 'happyEmo',
			  'sadEmo', 'containFirstPron','containSecPron','containThirdPron','numUpperCase', 'positiveWords', 'negativeWords', 
					   'numMentions', 'numHashtags', 'numUrls','rtCount' ,
					'slangWords','colonSymbol','pleasePresent' ,'WotValue', 'numQuesSymbol','numExclamSymbol', 'readabilityValue','Indegree',
					   'Harmonic' ,'AlexaCountry','AlexaDelta','AlexaPopularity', 'AlexaReach' , 'class']


	item_all_test = item_all_test.replace("?", 0)
	#item_all_test = item_all_test.fillna(item_all_test.mean())
	item_all_test = LinearReg(item_all_test, item_new)
	item_all_test = normalize(item_all_test)
	item_all_X_test = item_all_test.values[:,0:31]
	item_all_Y_test = item_all_test.values[:,31]

	for i in range(0,9):
		#item_split[i] = shuffle(item_split[i], random_state =70)
		model = pipeline(item_split[i])
		Ms.append(model)
		#cv_scores.append(model[1])
	#print Ms

	predscore_item = []
	prediction_values1, item_prediction_values, testing_val = [],[], []
	for m in Ms:
		item_all_predval = m.predict(item_all_test.values[:,0:31])
		item_prediction_values.append(item_all_predval)
		item_predscore = m.predict_proba(item_all_X_test)
		testing_val.append(item_all_Y_test)
		item_prediction_values.append(item_all_predval)
		prediction_values1.append(item_predscore)

	prediction_val = []
	item_combined_val = []
	abc = []
	for i in range(0,len(Ms)):
			column = []
			for j in range(0,len(item_all_Y_test)):
					if (prediction_values1[i][j][0]) > (prediction_values1[i][j][1]):
						column.append(prediction_values1[i][j][0])
					else:
						column.append(prediction_values1[i][j][1])
			prediction_val.append(column)

	for i in range(0,len(Ms)):
		item_column = []
		for j in range(0,len(item_all_X_test)):
			abc = item_prediction_values[i][j], testing_val[i][j], prediction_val[i][j]
			item_column.append(abc)
		item_combined_val.append(item_column)
				
	#pprint.pprint(item_combined_val)
	#print item_combined_val


	#pprint.pprint(item_prediction_values)
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
		a.append(b)
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

#print "Time taken:", time.time() - start_time
