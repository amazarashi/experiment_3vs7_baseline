import pickle
import glob
import os
import numpy as np
current = dir_path = os.path.dirname(os.path.realpath('__file__')) + "/"

import amaz_util as amaz_Util
import amaz_augumentation

class Cifar10(object):

    def __init__(self):
        self.cifar10url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.dlfile = "cifar-10-python.tar.gz"
        self.savepath = current + self.dlfile
        self.untarpath = current + "cifar-10-batches-py"
        self.train_files = ["data_batch_{0}".format(i) for i in range(1,6)]
        self.test_files = ["test_batch"]
        self.meta_files = ["batches.meta"]
        self.final_dataset_file = "cifar10.pkl"
        self.final_categorical_dataset_file = "cifar10_categorical.pkl"
        self.utility = amaz_Util.Utility()
        self.augumentation = amaz_augumentation.Augumentation()

    def downloader(self):
        allfiles_in_current = [path.replace(current,"") for path in glob.glob(current+"*")]

        """
         * 1 : judging to donwload or not
        """
        if self.dlfile in allfiles_in_current:
            print(self.dlfile + " is already existing..")
        else:
            #donwnload cifar10
            print("begin to Download Cifar10")
            self.utility.download_file_fromurl(self.cifar10url,self.savepath)
            print("ended to Download Cifar10")

        """
         * 2 : untar tar file
        """
        if self.untarpath.replace(current,"") in allfiles_in_current:
            print("already untared....")
        else:
            print("begin to untar")
            self.utility.untar_file(self.savepath)
            print("ended to untar")

        #load data for Train
        """
         * 3 : load data
        """
        if self.final_dataset_file.replace(current,"") in allfiles_in_current:
            print("already loaded....")
        else:
            print("load Train data")
            train_x = np.zeros((50000,3,32,32),dtype=np.float32)
            train_y = np.zeros((50000,),dtype=np.int32)

            for i,batch_file_path in enumerate(self.train_files):
                train_data = self.utility.unpickle(self.untarpath + "/" + batch_file_path)
                start = i*10000
                end = start + 10000
                train_x[start:end] = train_data["data"].reshape((10000,3,32,32))
                train_y[start:end] = np.array(train_data["labels"],dtype=np.int32)

            #load data for Test
            print("load Test data")
            test_x = np.zeros((10000,3,32,32),dtype=np.float32)
            test_y = np.zeros((10000,),dtype=np.int32)
            test_data = self.utility.unpickle(self.untarpath + "/" + self.test_files[0])
            test_x[:] = test_data["data"].reshape((10000,3,32,32))
            test_y[:] = np.array(test_data["labels"],dtype=np.int32)

            #load meta info of dataset
            print("load Meta Info")
            meta_data = self.utility.unpickle(self.untarpath + "/" + self.meta_files[0])
            meta = meta_data["label_names"]

            data = {}
            data["train_x"] = self.augumentation.Z_score(train_x)
            data["train_y"] = train_y
            data["test_x"] = self.augumentation.Z_score(test_x)
            data["test_y"] = test_y
            data["meta"] = meta

            #save on pkl file
            print("saving to pkl file ...")
            savepath = current + self.final_dataset_file
            self.utility.savepickle(data,savepath)
        print("data preparation was done ...")

    def loader(self):
        """
        with download check
        """
        self.downloader()
        data = self.utility.unpickle(current + self.final_dataset_file)
        return data

    def simpleLoader(self):
        """
        without download check
        """
        data = self.utility.unpickle(current + self.final_dataset_file)
        return data

    # for myselfs research
    def makeElseData(self):
        #label type:['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        data = self.utility.unpickle(current + self.final_dataset_file)
        train_x = data["train_x"]
        train_y = data["train_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]
        meta = data["meta"]

        category_base_data = {}
        for ind in range(len(meta)):
            category = meta[ind]
            target_train_y_index = np.where(train_y == ind)
            target_train_x = train_x[target_train_y_index]
            target_test_y_index = np.where(test_y == ind)
            target_test_x = test_x[target_test_y_index]
            categorical_data = {}
            categorical_data["train"] = target_train_x
            categorical_data["test"] = target_test_x
            category_base_data[category] = categorical_data
        category_base_data["meta"] = meta
        #save on pkl file
        print("saving to pkl file ...")
        savepath = current + self.final_categorical_dataset_file
        self.utility.savepickle(category_base_data,savepath)

    def categorical_loader(self):
        allfiles_in_current = [path.replace(current,"") for path in glob.glob(current+"*")]

        if self.final_categorical_dataset_file in allfiles_in_current:
            print(self.final_categorical_dataset_file + " is already existing..")
        else:
            self.makeElseData()
        data = self.utility.unpickle(current + self.final_categorical_dataset_file)

        return data
