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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import time
import pprint
from sklearn.metrics import f1_score
from userFeaturesClassification import userClassification
from itemFeaturesClassification import itemClassification
from itemFeaturesClassification import split_dataframe
from itemFeaturesClassification import LinearReg
from itemFeaturesClassification import item_testing_data
start_time = time.time()
finVal = itemClassification()
user_finVal = userClassification()
agreedVal = []
disagreedVal = []
for i in range(len(finVal)):
    column = []
    for j in range(len(finVal[i])):
        if finVal[i][j] == user_finVal[i][j]:
            column.append(finVal[i][j])
        else:
            column.append("disagreed")
    agreedVal.append(column)

for j in range(len(item_testing_data)):
    item_testing_data[j] = item_testing_data[j].set_index(np.array([i for i in agreedVal[j]]))
	
from collections import Counter
disagreed_dfs = []
for i in range(len(item_testing_data)):
    disagreed_dfs.append(item_testing_data[i].loc[['disagreed']])
#print len(disagreed_dfs)
agreed_dfs = []

for i in range(0, len(item_testing_data)):
    count_index = Counter(item_testing_data[i].index)
    if 'real' in count_index.keys():
        fake_df = item_testing_data[i].loc[['fake']]
        real_df = item_testing_data[i].loc[['real']]
        agreed_dfs.append(pd.concat([fake_df, real_df]))
    else:
        agreed_dfs.append(item_testing_data[i].loc[['fake']])
    

for i in range(0,len(item_testing_data)):
    print len(agreed_dfs[i])
    print len(disagreed_dfs[i])
    count_index = Counter(item_testing_data[i].index)
    print count_index
	

	
repredict = []
retest_df = []
retest_X = []
retest_Y = []
for i in range(0, len(disagreed_dfs)):
    retest_df.append(disagreed_dfs[i])
    retest_X.append(retest_df[i].values[:,0:31])
    retest_Y.append(retest_df[i].values[:,31])

for i in range(len(disagreed_dfs)):
    repredict.append(retrain_model.predict(retest_X[i]))

	
for i in range(len(agreedVal)):
    k = 0
    for j in range(len(agreedVal[i])):
        if agreedVal[i][j] == 'disagreed':
            agreedVal[i][j] = repredict[i][k]
            k = k + 1
			
acc_scores=[]

def calc_accuracy(fin_val):
    for i in range(0,len(fin_val)):
        acc_scores.append(accuracy_score(item_Y_test[i], agreedVal[i]))
    return acc_scores
acc = calc_accuracy(agreedVal)
print acc
def calc_fake_accuracy(agreedVal):
	accuracy_val_fake = []
	cmat_total = []
	cmat_val = []
	for i in range(len(agreedVal)):
		y_true = item_Y_test[i]
		y_pred = agreedVal[i]
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
	print accuracy_val_fake
	avg = sum(accuracy_val_fake)/len(accuracy_val_fake)
	print avg

