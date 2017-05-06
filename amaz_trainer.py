#coding : utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time

import amaz_sampling
import amaz_util
import amaz_sampling
import amaz_datashaping
import amaz_log
import amaz_augumentationCustom

sampling = amaz_sampling.Sampling()

class Trainer(object):

    def __init__(self,model=None,optimizer=None,dataset=None,epoch=300,batch=128,gpu=-1,dataaugumentation=amaz_augumentationCustom.Normalize32):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch
        self.train_x,self.train_y,self.test_x,self.test_y,self.meta = self.init_dataset()
        self.gpu = gpu
        self.check_gpu_status = self.check_gpu(self.gpu)
        self.xp = self.check_cupy(self.gpu)
        self.utility = amaz_util.Utility()
        self.datashaping = amaz_datashaping.DataShaping(self.xp)
        self.logger = amaz_log.Log()
        self.dataaugumentation = dataaugumentation

    def check_cupy(self,gpu):
        if gpu == -1:
            return np
        else:
            #cuda.get_device(gpu).use()
            self.model.to_gpu()
            return cuda.cupy

    def check_gpu(self, gpu):
        if gpu >= 0:
            #cuda.get_device(gpu).use()
            #self.to_gpu()
            return True
        return False

    def init_dataset(self):
        train_x = self.dataset["train_x"]
        train_y = self.dataset["train_y"]
        test_x = self.dataset["test_x"]
        test_y = self.dataset["test_y"]
        meta = self.dataset["meta"]
        return (train_x,train_y,test_x,test_y,meta)

    def train_one(self,epoch):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        train_x = self.train_x
        train_y = self.train_y
        meta = self.meta
        sum_loss = 0
        total_data_length = len(train_x)

        progress = self.utility.create_progressbar(int(total_data_length/batch),desc='train',stride=1)
        train_data_yeilder = sampling.random_sampling(int(total_data_length/batch),batch,total_data_length)
        #epoch,batch_size,data_length
        for i,indices in zip(progress,train_data_yeilder):
            model.cleargrads()
            x = train_x[indices]
            t = train_y[indices]

            DaX = []
            for img in x:
                da_x = self.dataaugumentation.train(img)
                DaX.append(da_x)

            x = self.datashaping.prepareinput(DaX,dtype=np.float32,volatile=False)
            t = self.datashaping.prepareinput(t,dtype=np.int32,volatile=False)

            y = model(x,train=True)
            loss = model.calc_loss(y,t)
            loss.backward()
            loss.to_cpu()
            sum_loss += loss.data * len(indices)
            del loss,x,t
            optimizer.update()

        ## LOGGING ME
        print("train mean loss : ",float(sum_loss) / total_data_length)
        self.logger.train_loss(epoch,sum_loss/len(train_y))
        print("######################")

    def test_one(self,epoch):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        test_x = self.test_x
        test_y = self.test_y
        meta = self.meta

        sum_loss = 0
        sum_accuracy = 0
        progress = self.utility.create_progressbar(int(len(test_x)),desc='test',stride=batch)
        for i in progress:
            x = test_x[i:i+batch]
            t = test_y[i:i+batch]

            DaX = []
            for img in x:
                da_x = self.dataaugumentation.test(img)
                DaX.append(da_x)

            x = self.datashaping.prepareinput(DaX,dtype=np.float32,volatile=True)
            t = self.datashaping.prepareinput(t,dtype=np.int32,volatile=True)

            y = model(x,train=False)
            loss = model.calc_loss(y,t)
            sum_loss += batch * loss.data
            sum_accuracy += F.accuracy(y,t).data * batch
            categorical_accuracy = model.accuracy_of_each_category(y,t)
            del loss,x,t

        ## LOGGING ME
        print("test mean loss : ",sum_loss/len(test_y))
        self.logger.test_loss(epoch,sum_loss/len(test_y))
        print("test mean accuracy : ", sum_accuracy/len(test_y))
        self.logger.accuracy(epoch,sum_accuracy/len(test_y))
        print("######################")

    def run(self):
        epoch = self.epoch
        model = self.model
        progressor = self.utility.create_progressbar(epoch,desc='epoch',stride=1,start=0)
        for i in progressor:
            self.train_one(i)
            self.optimizer.update_parameter(i)
            self.test_one(i)
            #DUMP Model pkl
            model.to_cpu()
            self.logger.save_model(model=model,epoch=i)
            if self.check_gpu_status:
                model.to_gpu()

        self.logger.finish_log()
