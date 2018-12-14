import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans

#### Dataset ####
dataset_in = np.loadtxt("vehicle.csv", delimiter=",", dtype=str)

def reshape_input(input_data):
    output = np.array([[input_data[0]]])
    for i in range(1, input_data.shape[0]):
        output = np.vstack((output, np.array([[input_data[i]]])))
    return output

X = reshape_input(dataset_in)
# To get a vector of labels: X.T[18]
# To get a single case vector: X[n].T
# To get a 2D array of the data without labels: X[1:,0,:-1]



#### Kmeans ####
def run(input_arr, ):
    kmeans = KMeans(n_clusters = 4, batch_size = 10) # initialization is performed using kmeans++
    predictions = kmeans.fit_predict(input_arr) # Returns a list of values in {0...3}
    labels = X.T[18][0]
    clusters = [indices(0), indices(1), indices(2), indices(3)]
    count = 1
    def indices(i):
        return np.where(predictions == i)

    for cluster in clusters:
        van = 0;    bus = 0
        saab = 0;   opel = 0
        for i in cluster[0]:
            if labels[i] == 'van':
                van += 1
            elif labels[i] == 'bus':
                bus += 1
            elif labels[i] == 'saab':
                saab += 1
            elif labels[i] == 'opel':
                opel += 1
        print("Cluster {}".format(count))
        print("van:", van)
        print("bus:", bus)
        print("saab:", saab)
        print("opel: {}\n".format(opel))
        count += 1



#### Testing ####
original_data = X[1:,0,:-1]

run(original_data)