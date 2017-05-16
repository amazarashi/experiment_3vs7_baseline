import urllib.request
import tarfile
from os import system
import sys
import six
import pickle
import numpy as np
from chainer import serializers
from chainer import cuda

import amaz_cifar10_dl
import darknet19
import amaz_augumentation
import amaz_augumentationCustom
import amaz_datashaping

from sklearn.cluster import KMeans

class KmeansProcess(object):

    def __init__(self):
        pass

    def __call__(self):
        pass

    def calc_categorical_centroid(self,features):
        km_model = KMeans(n_clusters=1, random_state=10).fit(features)

        #Centroid
        centroid = km_model.cluster_centers_[0]
        #sum of distance from centroid
        sum_distances = km_model.inertia_

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

    def updateCentroid(self,model,elseIndices):
        class_num = 10
        allInd = np.arange(class_num)
        elseInd = np.array(elseIndices)
        trainedInd = np.delete(allInd,elseInd)

        #prepare data
        dataset = amaz_cifar10_dl.Cifar10().categorical_loader()
        meta = np.array(dataset["meta"])
        else_meta = meta[elseInd]
        trained_meta = meta[trainedInd]

        #toolkit
        dataaugumentation = amaz_augumentationCustom.Normalize224
        datashaping = amaz_datashaping.DataShaping(cuda.cupy)

        #get centroid of each category
        maxdis_res = []
        for tm in trained_meta:
            labelname = tm
            ctgcalimgs = dataset[labelname]["train"]
            features = []
            print(labelname)
            print(len(ctgcalimgs))
            for i,img in enumerate(ctgcalimgs):
                x = [amaz_augumentation.Augumentation().Z_score(img)]
                da_x = [dataaugumentation.test(xx) for xx in x]
                xin = datashaping.prepareinput(da_x,dtype=cuda.cupy.float32,volatile=True)
                xin.to_gpu()
                feature = model.getFeature(xin,train=False)
                feature.to_cpu()
                print(labelname,":",i)
                features.append(feature.data[0])
            centroid,maxdis = amaz_kmeans.KmeansProcess().calc_categorical_centroid(np.array(features))
            maxdis_res.append([labelname,centroid,maxdis])
        return (trained_meta,maxdis_res)

    def calcElseScore(self):
        return
