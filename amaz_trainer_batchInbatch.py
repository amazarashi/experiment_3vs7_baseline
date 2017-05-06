#coding : utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import six
import pickle
from tqdm import tqdm
import amaz_sampling
import amaz_util
import amaz_sampling
import amaz_datashaping
import amaz_log
import amaz_augumentationCustom
import amaz_imagenet
import sys

sampling = amaz_sampling.Sampling()

xp = cuda.cupy

class Trainer(object):

    def __init__(self,model=None,batchinbatch=16,loadmodel=None,elseIndices=None,optimizer=None,dataset=None,epoch=300,batch=64,gpu=-1,dataaugumentation=amaz_augumentationCustom.Normalize32):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch
        self.elseIndices = elseIndices
        self.train_x,self.train_y,self.test_x,self.test_y,self.else_train_x,self.else_train_y,self.else_test_x,self.else_test_y,self.meta = self.init_dataset()
        self.gpu = gpu
        self.check_gpu_status = self.check_gpu(self.gpu)
        self.xp = self.check_cupy(self.gpu)
        self.utility = amaz_util.Utility()
        self.datashaping = amaz_datashaping.DataShaping(self.xp)
        self.logger = amaz_log.Log()
        self.dataaugumentation = dataaugumentation
        self.batchinbatch = batchinbatch
        self.loadmodel = loadmodel
        self.init_model()

    def check_cupy(self,gpu):
        if gpu == -1:
            return np
        else:
            #cuda.get_device(gpu).use()
            #self.model.to_gpu()
            return cuda.cupy

    def check_gpu(self, gpu):
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu()
            return True
        return False

    def init_model(self):
        if self.loadmodel is None:
            print('no model to load')
        else:
            print('loading ' + self.load_model)
            serializers.load_npz(load_model, self.model)
        self.check_gpu(self.gpu)

    def init_dataset(self):
        elseIndices = self.elseIndices
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        else_train_x = []
        else_train_y = []
        else_test_x = []
        else_test_y = []
        meta = self.dataset["meta"]
        for ind in range(len(meta)):
            category = meta[ind]
            categorical_data = self.dataset[category]
            train_data = categorical_data["train"]
            test_data = categorical_data["test"]
            if ind in elseIndices:
                else_train_x.extend(train_data)
                else_train_y += list(range(len(train_data)))
                else_test_x.extend(test_data)
                else_test_y += list(range(len(test_data)))
            else:
                train_x.extend(train_data)
                train_y += list(range(len(train_data)))
                test_x.extend(test_data)
                test_y += list(range(len(test_data)))
        return (np.array(train_x),
                np.array(train_y),
                np.array(test_x),
                np.array(test_y),
                np.array(else_train_x),
                np.array(else_train_y),
                np.array(else_test_x),
                np.array(else_test_y),
                np.array(meta))

    def train_one(self,epoch):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        meta = self.meta
        train_x = self.train_x
        train_y = self.train_y
        print(len(train_x))
        print(len(train_y))

        sum_loss = 0
        total_data_length = len(train_x)
        batch_in_batch_size = self.batchinbatch
        train_batch_devide = batch/batch_in_batch_size

        progress = self.utility.create_progressbar(int(total_data_length/batch),desc='train',stride=1)
        train_data_yeilder = sampling.random_sampling(int(total_data_length/batch),batch,total_data_length)
        #epoch,batch_size,data_length
        batch_in_batch_size = self.batchinbatch
        for i,indices in zip(progress,train_data_yeilder):
            model.cleargrads()
            print(indices)
            print(len(indices))
            print("#####")
            print("#####")
            print("#####")
            x = train_x[indices]
            t = train_y[indices]

            DaX = [self.dataaugumentation.train(img) for img in x]

            x = self.datashaping.prepareinput(DaX,dtype=np.float32,volatile=False)
            t = self.datashaping.prepareinput(t,dtype=np.int32,volatile=False)

            y = model(x,train=True)
            loss = model.calc_loss(y,t)
            loss.backward()
            loss.to_cpu()
            sum_loss += loss.data * len(indices)
            del loss,x,t
            optimizer.update()

        # for i,indices in zip(progress,train_data_yeilder):
        #     model.cleargrads()
        #     for ii in six.moves.range(0, len(indices), batch_in_batch_size):
        #         # print(ii)
        #         x = train_x[ii:ii + batch_in_batch_size]
        #         t = train_y[ii:ii + batch_in_batch_size]
        #         d_length = len(x)
        #
        #         DaX = [self.dataaugumentation.train(img) for img in x]
        #         x = self.datashaping.prepareinput(DaX,dtype=np.float32,volatile=False)
        #         t = self.datashaping.prepareinput(t,dtype=np.int32,volatile=False)
        #
        #         y = model(x,train=True)
        #         loss = model.calc_loss(y,t) / train_batch_devide
        #         print(i,ii,loss.data)
        #         loss.backward()
        #         loss.to_cpu()
        #         # print("loss",loss.data)
        #         # print("batch_in_batch_size",batch_in_batch_size)
        #         sum_loss += loss.data * d_length
        #         print(sum_loss)
        #         del loss,x,t
        #     optimizer.update()

        ## LOGGING ME
        print("train mean loss : ",float(sum_loss) / total_data_length)
        self.logger.train_loss(epoch,sum_loss/total_data_length)
        print("######################")


    def test_one(self,epoch):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        meta = self.meta
        test_x = self.test_x
        test_y = self.test_y

        sum_loss = 0
        sum_accuracy = 0
        batch_in_batch_size = self.batchinbatch

        progress = self.utility.create_progressbar(int(len(test_x)),desc='test',stride=batch)
        for i in progress:
            x = test_x[i:i+batch]
            t = test_y[i:i+batch]
            d_length = len(x)

            DaX = [self.dataaugumentation.train(img) for img in x]
            x = self.datashaping.prepareinput(DaX,dtype=np.float32,volatile=True)
            t = self.datashaping.prepareinput(t,dtype=np.int32,volatile=True)

            y = model(x,train=False)
            loss = model.calc_loss(y,t)
            sum_loss += d_length * loss.data
            sum_accuracy += F.accuracy(y,t).data * d_length
            #categorical_accuracy = model.accuracy_of_each_category(y,t)
            del loss,x,t

        ## LOGGING ME
        print("test mean loss : ",sum_loss/len(self.test_x))
        self.logger.test_loss(epoch,sum_loss/len(self.test_x))
        print("test mean accuracy : ", sum_accuracy/len(self.test_x))
        self.logger.accuracy(epoch,sum_accuracy/len(self.test_x))
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
