import urllib.request
import tarfile
from os import system
import os
import sys
import six
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Log(object):

    def __init__(self):
        self.name = "log"
        self.model_save_path = "./model"
        self.log_save_path = "./log"
        self.train_loss_path = "log/train_loss.txt"
        self.test_loss_path = "log/test_loss.txt"
        self.accuracy_path = "log/accuracy.txt"
        self.loss_result_path = "log/loss.png"
        self.accuracy_result_path = "log/accuracy.png"
        self.init = self.init_log_env()
        self.train_loss_fp = open(self.train_loss_path, "w")
        self.test_loss_fp = open(self.test_loss_path, "w")
        self.accuracy_fp = open(self.accuracy_path, "w")
        self.init_log_file = self.init_log_file()

    def init_log_env(self):
        if os.path.exists(self.model_save_path) == False:
            os.mkdir("model")
        if os.path.exists(self.log_save_path) == False:
            os.mkdir("log")
        return

    def init_log_file(self):
        self.train_loss_fp.write("epoch,train_loss\n")
        self.test_loss_fp.write("epoch,test_loss\n")
        self.accuracy_fp.write("epoch,accuracy\n")
        return

    def finish_log(self):
        self.train_loss_fp.close()
        self.test_loss_fp.close()
        self.accuracy_fp.close()
        return

    def save_model(self,model,epoch):
        pickle.dump(model, open(self.model_save_path+"/model_{0}.pkl".format(str(epoch)), "wb"), -1)
        return

    def train_loss(self,epoch,loss):
        self.train_loss_fp.write("%d,%f\n" % (epoch, loss))
        self.train_loss_fp.flush()
        return

    def test_loss(self,epoch,loss):
        self.test_loss_fp.write("%d,%f\n" % (epoch, loss))
        self.test_loss_fp.flush()
        self.plt_loss()
        return

    def accuracy(self,epoch,accuracy):
        self.accuracy_fp.write("%d,%f\n" % (epoch, accuracy))
        self.accuracy_fp.flush()
        self.plt_accuracy()
        return

    def plt_loss(self):
        aixisAry,train_lossAry = self.load_plt_data(self.train_loss_path)
        aixisAry,test_lossAry = self.load_plt_data(self.test_loss_path)
        plt.clf()
        plt.plot(aixisAry,train_lossAry, label='train')
        plt.plot(aixisAry,test_lossAry, label='test')
        plt.title('loss')
        plt.legend()
        plt.draw()
        plt.savefig(self.loss_result_path)
        return

    def plt_accuracy(self):
        aixisAry,accuracyAry = self.load_plt_data(self.accuracy_path)
        plt.clf()
        plt.plot(aixisAry,accuracyAry, label='test')
        plt.title('accuracy')
        plt.legend()
        plt.draw()
        plt.savefig(self.accuracy_result_path)
        return

    def load_plt_data(self,filepath):
        data = pd.read_csv(filepath, sep=",", header = None)
        data.columns = ["epoch","value"]
        df = pd.DataFrame(data)
        df.drop(df.index[0])
        valuesAry = [value[1] for value in df[1:].values]
        axissize = len(valuesAry)
        aixisAry = np.arange(axissize)
        return (aixisAry,valuesAry)
