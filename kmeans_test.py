import random
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import PCA
import import_data

#### Dataset ####

X = import_data.X
labels = X.T[21]
    # To get a vector of features: X.T[n,1:]
    # To get a single case vector: X[n].T
    # To get a 2D array of the data without labels: X[1:,0,:-1]

# Reduce the dimensionality of the dataset using Principal Component Analysis


#### Kmeans ####
def run(input_arr, nc, bs):
    input_reduced = PCA(n_components = 2).fit_transform(input_arr)
    kmeans = KMeans(n_clusters = nc, batch_size = bs) # initialization is performed using kmeans++
    kmeans.fit(input_reduced)
    predictions = kmeans.predict(input_reduced) # Returns a list of values in {0...3}

    centroids = kmeans.cluster_centers_
    k_labels = kmeans.labels_
    colour_map = {0:"r", 1:"k"}
    label_colour = [colour_map[l] for l in labels]

    def makeKMGraph(input_reduced, labels, name):
        fig, ax = plt.subplots()
        for i in range(len(input_reduced)):
            if labels[i] == 0:
                plt.scatter(input_reduced[i,0], input_reduced[i,1], color="r", marker="$0$")
            if labels[i] == 1:
                plt.scatter(input_reduced[i,0], input_reduced[i,1], color="b", marker="$1$")
        plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=5,
                color='k', zorder=10)

        x_min = input_reduced[:, 0].min() - 1
        x_max = input_reduced[:, 0].max() + 1
        y_min = input_reduced[:, 1].min() - 1
        y_max = input_reduced[:, 1].max() + 1

        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)

        plt.title("Kmeans on MNIST dataset, 2 clusters")

        plt.savefig('graphs/{}.png'.format(name))
        plt.show()

    makeKMGraph(input_reduced, labels, "K-means clustering of edible and poisonous mushrooms")

    # Print clusters
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
clusters, predictions = run(X, 2, 250)

#### Plotting the data ####
