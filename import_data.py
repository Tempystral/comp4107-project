import numpy as np

csv = "mushroom.csv" # Dataset retrieved from https://www.openml.org/d/24
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

X = make_numeric(dataset_in[1:])