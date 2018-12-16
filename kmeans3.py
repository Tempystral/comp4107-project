import random
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans
import import_data

#### Dataset ####
#dataset_in = np.loadtxt("phonemes.csv", delimiter=",", dtype=str)

def reshape_input(input_data):
    output = np.array([[input_data[0]]])
    for i in range(1, input_data.shape[0]):
        output = np.vstack((output, np.array([[input_data[i]]])))
    return output

#def regularize(input_data):


X = import_data.X #reshape_input(dataset_in)
# To get a vector of features: X.T[n,1:]
# To get a single case vector: X[n].T
# To get a 2D array of the data without labels: X[1:,0,:-1]

X_extracted = import_data.X[:,import_data.featureLists] #extracct some features from all


#### Kmeans ####
def run(input_arr, nc, bs):
    kmeans = KMeans(n_clusters = nc, batch_size = bs) # initialization is performed using kmeans++
    predictions = kmeans.fit_predict(input_arr) # Returns a list of values in {0...3}
    labels = X.T[21]
    count = 1
    def indices(i):
        return np.where(predictions == i)
    clusters = [indices(0), indices(1)]

    for cluster in clusters:
        poisonous = 0; edible = 0
        for i in cluster[0]:
            if labels[i] == 0:
                poisonous += 1
            elif labels[i] == 1:
                edible += 1
        print("Cluster {}".format(count))
        print("poisonous: {}".format(poisonous))
        print("edible: {}\n".format(edible))
        count += 1
    return clusters


#### Testing ####
#original_data = X[1:,0,:-1]
#single_vector = X[1:,0,5:8]
#U,s,V = la.svd(original_data)

clusters = run(X, 2, 250)
# print("After feature selections ... \n")
# clusters = run(X_extracted, 2, 250)
