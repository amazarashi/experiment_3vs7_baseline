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
    modelpath = "trained/model_265.pkl"
    model = darknet19.Darknet19(category_num=class_num-len(elseInd))
    model.to_gpu()
    serializers.load_npz(modelpath, model)
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
            xin = datashaping.prepareinput(da_x,dtype=np.float32,volatile=True)
            feature = model.getFeature(xin,train=False)
            feature.to_cpu()
            print(labelname,":",i)
            features.append(feature.data[0])
        centroid,maxdis = amaz_kmeans.KmeansProcess().calc_categorical_centroid(np.array(features))
        maxdis_res.append([labelname,centroid,maxdis])

    #debug
    for res in maxdis_res:
        labelname,centroid,maxdis = res
        print(labelname,":",maxdis)

    else_judge = 0
    nonelse_judge = 0
    log = {}
    for em in else_meta:
        labelname = tm
        ctgcalimgs = dataset[labelname]["test"]
        features = []
        for i,img in enumerate(ctgcalimgs):
            x = [amaz_augumentation.Augumentation().Z_score(img)]
            da_x = [dataaugumentation.test(xx) for xx in x]
            xin = datashaping.prepareinput(da_x,dtype=np.float32,volatile=True)
            feature = model.getFeature(xin,train=False)
            feature.to_cpu()
            feature = feature.data[0]
            # for f in feature:
            elseStatus = False
            for res in maxdis_res:
                labelname,centroid,maxdis = res
                print(labelname,":",maxdis)
                distance = amaz_kmeans.KmeansProcess().calc_distance_2point(centroid,feature)
                print("distances:",distance)
                print(centroid[:30])
                print(feature[:30])
                if distance < maxdis:
                    elseStatus = True
                    nonelse_judge += 1
                    if labelname in log.keys():
                        log[labelname] = int(log[labelname]) + 1
                    else:
                        log[labelname] = 1
                    print("judged as :"+labelname)
            if elseStatus == False:
                else_judge += 1
                print("ELSE")
            elseStatus = False

    print("else:",else_judge)
    print("non:",nonelse_judge)
    print(log)
