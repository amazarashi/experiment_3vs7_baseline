import pickle
import glob
import os
import numpy as np
current = dir_path = os.path.dirname(os.path.realpath('__file__')) + "/"

import amaz_util as amaz_Util
import amaz_augumentation

from os import system
from PIL import Image
from bs4 import BeautifulSoup as Soup
import cv2

from multiprocessing import Pool


class ImageNet(object):

    def __init__(self):
        self.trainImageBasePath = "/home/codenext2/Downloads/ILSVRC/"
        self.annotationsPath = self.trainImageBasePath + "Annotations/CLS-LOC/"
        self.dataPath = self.trainImageBasePath + "Data/CLS-LOC/"
        self.imgSetsPath = self.trainImageBasePath + "ImageSets/CLS-LOC/"
        self.final_dataset_file = "imagenet.pkl"
        self.utility = amaz_Util.Utility()
        self.category_num = 0
        self.meta = []

    def loader(self):

        allfiles_in_current = [path for path in glob.glob("*")]

        if self.final_dataset_file in allfiles_in_current:
            print(self.final_dataset_file + " is already existing..")
        else:
            self.arrangement()

        return

    def simpleLoader(self):
        """
        without download check
        """
        data = self.utility.unpickle(current + self.final_dataset_file)
        return data

    def arrangement(self):
        #load data for Train
        """
         * load data
        """
        #get all categories meta
        alllist = os.listdir(self.dataPath + "train/")
        metalist = alllist
        self.meta = metalist
        category_num = len(metalist)
        print("category_num: ",category_num)
        self.category_num = category_num

        #get annotation info
        trainImageSetPath = self.imgSetsPath + "train_loc.txt"
        valImageSetPath = self.imgSetsPath + "val.txt"

        trainImgs = open(trainImageSetPath,"r")
        trainImgs = trainImgs.readlines()
        trainImgs = [info.split()[0] for info in trainImgs]
        valImgs = open(valImageSetPath,"r")
        valImgs = valImgs.readlines()
        valImgs = [info.split()[0] for info in valImgs]
        print("trainLength:",len(trainImgs))
        print("valLength:",len(valImgs))

        # print("loading traindata ,,,,,,,")
        # trainData = {}
        # count_trian = 0
        # for trainimg in trainImgs:
        #     imgpath = self.dataPath + "train/" + trainimg + ".JPEG"
        #     annotationpath = self.annotationsPath + "train/" + trainimg + ".xml"
        #     label = self.loadXML(annotationpath)
        #     print(count_trian)
        #     print("imgpath:",imgpath)
        #     print("label:",label)
        #     trainData[trainimg] = {"imgpath":imgpath,"label":label,"label_index":self.ctg_ind(label)}
        #     count_trian += 1
        # print("train length:",count_trian)
        #
        # print("loading valdata ,,,,,,,")
        # valData = {}
        # count_val = 0
        # for valimg in valImgs:
        #     imgpath = self.dataPath + "val/" + valimg + ".JPEG"
        #     annotationpath = self.annotationsPath + "val/" + valimg + ".xml"
        #     label = self.loadXML(annotationpath)
        #     trainData[valimg] = {"imgpath":imgpath,"label":label,"label_index":self.ctg_ind(label)}
        #     count_val += 1
        # print("train length:",count_trian)

        res = {}
        res["train_key"] = trainImgs
        res["val_key"] = valImgs
        res["meta"] = metalist

        #save on pkl file
        print("saving to pkl file ...")
        savepath = self.final_dataset_file
        self.utility.savepickle(res,savepath)
        print("data preparation was done ...")
        return self.category_num

    def ctg_ind(self,ctgname,meta):
        meta = np.array(meta)
        ind = np.where(meta==ctgname)[0][0]
        return ind

    def loadXML(self,filepath):
        d = open(filepath).read()
        soup = Soup(d,"lxml")
        label = soup.find("name").text
        return label

    def loadImgs(self,imgpath):
        img = Image.open(imgpath)
        origshapetype = len(np.asarray(img).shape)
        if origshapetype == 2:
            img = cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
        transfromedImg = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
        resimg = amaz_augumentation.Augumentation().Z_score(transfromedImg)[:3]
        # print(imgpath)
        # print(resimg.shape)
        return resimg

    def loadImageDataFromKey(self,sampled_key_lists,dataKeyList,train_or_test):
        if train_or_test == "train":
            batchsize = len(sampled_key_lists)
            targetKeys = dataKeyList[sampled_key_lists]
        elif train_or_test == "val":
            batchsize = len(dataKeyList)
            targetKeys = dataKeyList[sampled_key_lists]

        with Pool(8) as p:
            imgdatas = p.map(self.loadImgs, [self.dataPath + train_or_test+ "/" + key + ".JPEG" for key in targetKeys])

        return imgdatas

    def loadImageAnnotationsFromKey(self,sampled_key_lists,dataKeyList,meta,annotation_filepath,train_or_test):
        d = open(annotation_filepath,"rb")
        dd = pickle.load(d)
        d.close()
        annotations = []
        if train_or_test == "train":
            targetKeys = dataKeyList[sampled_key_lists]
        elif train_or_test == "val":
            targetKeys = dataKeyList[sampled_key_lists]

        t = []
        for key in targetKeys:
            annotationpath = self.annotationsPath + train_or_test + "/" + key + ".xml"
            label = self.loadXML(annotationpath)
            label_ind = self.ctg_ind(label,meta)
            t.append(label_ind)
        return np.array(t)
