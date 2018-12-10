import random
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans

def load_data():
    datas=[]
    unlabeled_data=[]
    with open('vehicle.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            int_data = map(int, line[:18])          # convert string to int
            unlabeled_data.append(int_data)         # data without label
            temp = np.asarray(int_data)
            data = (line[18],temp)                  # data with label
            datas.append(data)

    return datas,unlabeled_data


def elbow_plot(data, maxK=10, seed_centroids=None):
    sse = {}
    for k in range(1, maxK):
        print("k: ", k)
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    return

def init_centroids(labelled_data,k):
    return map(lambda x: x[1], random.sample(labelled_data,k))

def sum_cluster(labelled_cluster):
    # assumes len(cluster) > 0
    sum_ = labelled_cluster[0][1].copy()
    for (label,vector) in labelled_cluster[1:]:
        sum_ += vector
    return sum_

def mean_cluster(labelled_cluster):
    sum_of_points = sum_cluster(labelled_cluster)
    mean_of_points = sum_of_points * (1.0 / len(labelled_cluster))
    return mean_of_points

def form_clusters(labelled_data, unlabelled_centroids):
    # enumerate because centroids are arrays which are unhashable
    centroids_indices = range(len(unlabelled_centroids))

    # initialize an empty list for each centroid. The list will
    # contain all the datapoints that are closer to that centroid
    # than to any other. That list is the cluster of that centroid.
    clusters = {c: [] for c in centroids_indices}

    for (label,Xi) in labelled_data:
        # for each datapoint, pick the closest centroid.
        smallest_distance = float("inf")
        for cj_index in centroids_indices:
            cj = unlabelled_centroids[cj_index]
            distance = np.linalg.norm(Xi - cj)
            if distance < smallest_distance:
                closest_centroid_index = cj_index
                smallest_distance = distance
        # allocate that datapoint to the cluster of that centroid.
        clusters[closest_centroid_index].append((label,Xi))
    return clusters.values()

def move_centroids(labelled_clusters):
    new_centroids = []
    for cluster in labelled_clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids

def repeat_until_convergence(labelled_data, labelled_clusters, unlabelled_centroids):
    previous_max_difference = 0
    while True:
        unlabelled_old_centroids = unlabelled_centroids
        unlabelled_centroids = move_centroids(labelled_clusters)
        labelled_clusters = form_clusters(labelled_data, unlabelled_centroids)
        # keep old_clusters and clusters so we can get the maximum difference
        # between centroid positions every time.
        differences = map(lambda a, b: np.linalg.norm(a-b),unlabelled_old_centroids,unlabelled_centroids)
        max_difference = max(differences)
        difference_change = abs((max_difference-previous_max_difference)/np.mean([previous_max_difference,max_difference])) * 100
        previous_max_difference = max_difference
        # difference change is nan once the list of differences is all zeroes.
        if np.isnan(difference_change):
            break
    return labelled_clusters, unlabelled_centroids

def cluster(labelled_data, k):
    centroids = init_centroids(labelled_data, k)
    clusters = form_clusters(labelled_data, centroids)
    final_clusters, final_centroids = repeat_until_convergence(labelled_data, clusters, centroids)
    return final_clusters, final_centroids

# elbow_plot(np.asarray(load_data()[1]))

result = cluster(load_data()[0],4)
for i in result[0]:
    van=0
    bus=0
    saab=0
    opel=0
    print("THE NUMBER OF POINTS IN THIS CLUSTER IS "+repr(len(i)))
    for j in i:
        if j[0] == 'van':
            van+=1
        elif j[0] == 'bus':
            bus+=1
        elif j[0] == 'saab':
            saab+=1
        elif j[0] == 'opel':
            opel+=1
    print("van: "+repr(van))
    print("bus: "+repr(bus))
    print("saab: "+repr(saab))
    print("opel: "+repr(opel))
