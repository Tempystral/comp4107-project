import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import heapq

csv = "mushroom.csv"
ignore = "?"

#### Dataset ####
dataset_in = np.loadtxt(csv, delimiter=",", dtype=str)

'''def reshape_input(input_arr):
    output = np.array([[input_arr[0]]])
    for i in range(1, input_arr.shape[0]):
        output = np.vstack((output, np.array([[input_arr[i]]])))
    return output'''

def make_numeric(input_arr):
	# transpose dataset
	arr = input_arr.T
	newTable = np.zeros(shape=(arr[0].shape[0]))
	for i in range(arr.shape[0]):
		if i == 10:
			continue	# This skips feature 11: stalk-root, which has missing data
		feature_map = {}
		count = 0
		feature = arr[i]
		newRow = np.zeros(shape=(feature.shape[0]))
		for x in range(feature.shape[0]):
			if feature[x] not in feature_map:	# Add entry to the map of features if it isn't there already
				feature_map[feature[x]] = count
				count += 1
			newRow[x] = feature_map[feature[x]]
		newTable = np.vstack((newTable, newRow))
	return newTable[1:].T

def getLables(input_arr):
    label_arr = input_arr[:,22]
    oneHot_arr = np.zeros(2)
    for i in label_arr:
        if i=='p':
            temp = np.asarray([1,0])
            oneHot_arr = np.vstack((oneHot_arr,temp))
        elif i=='e':
            temp = np.asarray([0,1])
            oneHot_arr = np.vstack((oneHot_arr,temp))
        else:
            print("ERROR")
    return oneHot_arr[1:]

def featureSelection(x,y):
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(x,y)
    featureRankingList = fit.scores_
    a = heapq.nlargest(14, featureRankingList)      # Select 14 most related features
    listOfFeatureIndex = []
    for i in a:
        listOfFeatureIndex.append(featureRankingList.tolist().index(i))
    return listOfFeatureIndex

Y = getLables(dataset_in[1:])
X = make_numeric(dataset_in[1:])
featureLists = featureSelection(X,Y)
