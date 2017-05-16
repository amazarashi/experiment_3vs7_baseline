import urllib.request
import tarfile
from os import system
import sys
import six
import pickle
import numpy as np

from sklearn.cluster import KMeans

class KmeansProcess(object):

    def __init__(self):
        pass

    def __call__(self):
        pass

    def calc_categorical_centroid(self,features):
        print(features.shape)
        print(type(features))
        km_model = KMeans(n_clusters=1, random_state=10).fit(features)

        #Centroid
        centroid = km_model.cluster_centers_[0]
        #sum of distance from centroid
        sum_distances = kmeans_model.inertia_

        maxdis = 0.
        maxdis_ind = 0
        for i,feature in enumerate(features):
            dis = self.calc_distance_2point(centroid,feature)
            if dis > maxdis:
                maxdis = dis
                maxdis_ind = i
        return (centroid,maxdis)

    def calc_distance_2point(self,pt1,pt2):
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        distance = np.sum((pt1 - pt2) ** 2 / 2)
        return distance
