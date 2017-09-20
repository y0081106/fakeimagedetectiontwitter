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
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import pprint
from userFeaturesClassification import userClassification
from itemFeaturesClassification import itemClassification
start_time = time.time()
finVal = itemClassification()
#print finval
user_finVal = userClassification()
agreedVal = []
disagreedVal = []
for i in range(0, len(finVal)):
    if finVal[i] == user_finVal[i]:
        agreedVal.append(finVal[i])
        #agcount =agcount+1
    else:
        agreedVal.append("disagreed")
#print agreedVal
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

retrain_set , retest_set =[],[]
for i in range(0, len(finVal)):
    if agreedVal[i] != "disagreed":
        retrain_set.append(item_all_test.values[i])
        
    if agreedVal[i] == "disagreed":
        retest_set.append(item_all_test.values[i])
        
retrain_df = pd.DataFrame(retrain_set)
retest_df = pd.DataFrame(retest_set)
retrain_df.columns = ['tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol', 'externLinkPresent','numberNouns', 'happyEmo',
          'sadEmo', 'containFirstPron','containSecPron','containThirdPron','numUpperCase', 'positiveWords', 'negativeWords', 
                   'numMentions', 'numHashtags', 'numUrls','rtCount' ,
                'slangWords','colonSymbol','pleasePresent' ,'WotValue', 'numQuesSymbol','numExclamSymbol', 'readabilityValue','Indegree',
                   'Harmonic' ,'AlexaCountry','AlexaDelta','AlexaPopularity', 'AlexaReach' , 'class']
#Concatenating data from previous training set and agreed set
dataframes =[item, retrain_df]
totalTrainingSet = pd.concat(dataframes)
totalTrainingSet = normalize(totalTrainingSet)
   
def retrainPipeline(df):
    X_retrain = df.values[:,0:31]
    Y_retrain = df.values[:,31]
    num_trees = 100
    rfc = RandomForestClassifier(n_estimators=num_trees, max_features= 1,random_state = 84, min_samples_split = 40,max_leaf_nodes = 15, max_depth = 5,min_samples_leaf = 3)
    retrain_model = rfc.fit(X_retrain, Y_retrain)
    return retrain_model
                        
model = retrainPipeline(totalTrainingSet)
repredict = []
retest_X = retest_df.values[:,0:31]
retest_Y = retest_df.values[:,31]
repredict.append(model.predict(retest_X))
print repredict
print agreedVal
print "Time taken:", time.time() - start_time
#print retest_Y

#repredict = model.predict(retest_X)
#print repredict
