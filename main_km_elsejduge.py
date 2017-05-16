from chainer import serializers
from chainer import cuda
import urllib.request
import tarfile
from os import system
import sys
import six
import pickle
import numpy as np

import amaz_cifar10_dl
import darknet19
import amaz_augumentation
import amaz_augumentationCustom
import amaz_datashaping
import amaz_kmeans

if __name__ == "__main__":
    #load dataset

    class_num = 10
    allInd = np.arange(class_num)
    elseInd = np.array([0,1,2])
    trainedInd = np.delete(allInd,elseInd)
    #prepare data
    dataset = amaz_cifar10_dl.Cifar10().categorical_loader()
    meta = np.array(dataset["meta"])
    else_meta = meta[elseInd]
    trained_meta = meta[trainedInd]
    #prepare model
    modelpath = "trained/model_40.pkl"
    model = darknet19.Darknet19(category_num=class_num-len(elseInd))
    model.to_gpu()
    serializers.load_npz(modelpath, model)
    #toolkit
    dataaugumentation = amaz_augumentationCustom.Normalize224
    np = cuda.cupy
    datashaping = amaz_datashaping.DataShaping(np)

    #get centroid of each category
    maxdis_res = []
    for tm in trained_meta:
        labelname = tm
        ctgcalimgs = dataset[labelname]["train"]
        features = []
        print(labelname)
        print(len(ctgcalimgs))
        for i,img in enumerate(ctgcalimgs):
            x = amaz_augumentation.Augumentation().Z_score(img)
            da_x = dataaugumentation.test(x)
            xin = datashaping.prepareinput([da_x],dtype=np.float32,volatile=True)
            featre = model.getFeature(xin,train=False)
            features.append(featre)
            print(i)

        centroid,maxdis = amaz_kmeans.KmeansProcess.calc_categorical_centroid(np.array(features.data))
        print(centroid,maxdis)
        maxdis_res.append([labelname,centroid,maxdis])

    # for em in else_meta:
    #     labelname = tm
    #     ctgcalimgs = dataset[labelname]["train"]
    #     features = []
    #     for i,img in enumerate(ctgcalimgs):
    #         x = amaz_augumentation.Augumentation().Z_score(img)
    #         da_x = dataaugumentation.test(x)
    #         xin = datashaping.prepareinput([da_x],dtype=np.float32,volatile=True)
    #         featre = model.getFeature(xin,train=False)
