import numpy as np
import pandas as pd
import pickle

from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import userFeaturesClassification
import itemFeaturesClassification

with open('item_testing_data.pkl', 'rb') as f:
    item_testing_data = pickle.load(f)

with open('item_Y_test.pkl', 'rb') as f:
    item_Y_test = pickle.load(f)

def retrainPipeline(df):
    X_retrain = df.values[:, 0:31]
    Y_retrain = df.values[:, 31]
    num_trees = 100
    rfc = RandomForestClassifier(n_estimators=num_trees, random_state=0)
    retrain_model = rfc.fit(X_retrain, Y_retrain)
    return retrain_model


def agreement(finVal, user_finVal):
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
    return agreedVal


def retraining(agreedVal):

    for j in range(len(item_testing_data)):
        item_testing_data[j] = item_testing_data[j].set_index(np.array([i for i in agreedVal[j]]))
    disagreed_dfs = []
    # append disagreed instances to disagreed_df
    for i in range(len(item_testing_data)):
        disagreed_dfs.append(item_testing_data[i].loc[['disagreed']])
    agreed_df = []
    agreed_dfs = []
    # append agreed instances to greed_df
    for i in range(0, len(item_testing_data)):
        count_index = Counter(item_testing_data[i].index)
        if 'real' in count_index.keys():
            fake_df = item_testing_data[i].loc[['fake']]
            real_df = item_testing_data[i].loc[['real']]
            agreed_dfs.append(pd.concat([fake_df, real_df]))
        else:
            agreed_dfs.append(item_testing_data[i].loc[['fake']])
            # concat all agreed_dfs into one
    agreed_df = pd.concat(agreed_dfs)
    retest_df = []
    retest_X = []
    retest_Y = []
    repredict = []
    # prepare retest data
    for i in range(0, len(disagreed_dfs)):
        retest_df.append(disagreed_dfs[i])
        retest_X.append(retest_df[i].values[:, 0:31])
        retest_Y.append(retest_df[i].values[:, 31])

    # retrain CL Classifier
    retrain_model = retrainPipeline(agreed_df)
    # predict retest samples
    for i in range(len(disagreed_dfs)):
        repredict.append(retrain_model.predict(retest_X[i]))
    # change agreedVal with repredicted samples
    for i in range(len(agreedVal)):
        k = 0
        for j in range(len(agreedVal[i])):
            if agreedVal[i][j] == 'disagreed':
                agreedVal[i][j] = repredict[i][k]
                k = k + 1

    return agreedVal


def calc_accuracy(agreedVal):
    acc_scores = []
    accuracy_val_fake = []
    cmat_total = []
    cmat_val = []
    for i in range(len(agreedVal)):
        y_true = item_Y_test[i]
        y_pred = agreedVal[i]
        cmat = confusion_matrix(y_true, y_pred)
        # print cmat
        cmat_val.append(cmat[0][0])
        if len(cmat[0] > 1):
            cmat_sum = 0
            for i in range(len(cmat)):
                cmat_sum = cmat_sum + cmat[0][i]
            cmat_total.append(cmat_sum)
        else:
            cmat_total.append(cmat[0])
    for i in range(len(cmat_val)):
        accr = float(cmat_val[i]) / cmat_total[i]
        accuracy_val_fake.append(float(accr))
    return accuracy_val_fake


# avg = sum(accuracy_val_fake)/len(accuracy_val_fake)
# return avg
def main():
	# predicted values from Tweet Features Classifier
	print "get finVal"
	with open('item_fin_val.pkl', 'rb') as f:
		finVal = pickle.load(f)
	# finVal = itemFeaturesClassification.final_pred()
	# predicted values from User Features Classifier
	print "get user_finVal"
	with open('user_fin_val.pkl', 'rb') as f:
		user_finVal = pickle.load(f)
	# user_finVal = userFeaturesClassification.final_pred()
	# Check for agreement of TF and UF Classifiers
	print "get agreed values"
	agreed_values = agreement(finVal, user_finVal)
	# Retrain the disagreed values
	print "retrain"
	agreed_values = retraining(agreed_values)
	print "average accuracy"
	accu = calc_accuracy(agreed_values)
	avg = sum(accu)/len(accu)
	print avg


if __name__ == '__main__':
    main()