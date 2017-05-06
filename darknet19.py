import chainer
import chainer.functions as F
import chainer.links as L
import skimage.io as io
import numpy as np
from chainer import utils
from tqdm import tqdm
import sys

class darkModule(chainer.Chain):

    def __init__(self,in_size,out_size,k_size=3,stride=1,pad=1):
        super(darkModule,self).__init__(
            conv = L.Convolution2D(in_size,out_size,k_size,stride=stride,pad=pad,nobias=True),
            bn = L.BatchNormalization(out_size,use_beta=False,eps=1e-5),
            bias = L.Bias(shape=(out_size,))
        )

    def __call__(self,x,train=False,finetune=False):
        h = self.conv(x)
        h = self.bn(h,test=not train,finetune=finetune)
        h = self.bias(h)
        h = F.leaky_relu(h,slope=0.1)
        return h

class Darknet19(chainer.Chain):

    def __init__(self,category_num=10):
        super(Darknet19,self).__init__(
            dark1 = darkModule(3,32,3,stride=1,pad=1),
            dark2 = darkModule(32,64,3,stride=1,pad=1),
            dark3 = darkModule(64,128,3,stride=1,pad=1),
            dark4 = darkModule(128,64,1,stride=1,pad=0),
            dark5 = darkModule(64,128,3,stride=1,pad=1),
            dark6 = darkModule(128,256,3,stride=1,pad=1),
            dark7 = darkModule(256,128,1,stride=1,pad=0),
            dark8 = darkModule(128,256,3,stride=1,pad=1),
            dark9 = darkModule(256,512,3,stride=1,pad=1),
            dark10 = darkModule(512,256,1,stride=1,pad=0),
            dark11 = darkModule(256,512,3,stride=1,pad=1),
            dark12 = darkModule(512,256,3,stride=1,pad=1),
            dark13 = darkModule(256,512,3,stride=1,pad=1),
            dark14 = darkModule(512,1024,3,stride=1,pad=1),
            dark15 = darkModule(1024,512,1,stride=1,pad=0),
            dark16 = darkModule(512,1024,3,stride=1,pad=1),
            dark17 = darkModule(1024,512,1,stride=1,pad=0),
            dark18 = darkModule(512,1024,3,stride=1,pad=1),
            conv19 = L.Convolution2D(1024,category_num,1,stride=1,pad=0)
        )

    def __call__(self,x,train=True):
        #x = chainer.Variable(x)
        h = self.dark1(x,train=train)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = self.dark2(h,train=train)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = self.dark3(h,train=train)
        h = self.dark4(h,train=train)
        h = self.dark5(h,train=train)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = self.dark6(h,train=train)
        h = self.dark7(h,train=train)
        h = self.dark8(h,train=train)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = self.dark9(h,train=train)
        h = self.dark10(h,train=train)
        h = self.dark11(h,train=train)
        h = self.dark12(h,train=train)
        h = self.dark13(h,train=train)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = self.dark14(h,train=train)
        h = self.dark15(h,train=train)
        h = self.dark16(h,train=train)
        h = self.dark17(h,train=train)
        h = self.dark18(h,train=train)
        h = self.conv19(h)
        num,categories,y,x = h.data.shape
        #average pool over (y,x) area
        h = F.average_pooling_2d(h,(y,x))
        # if categories = n
        # [num1[0,1,,,n],num2[0,1,2,,,n],,,]
        h = F.reshape(h,(num,categories))
        return h

    def calc_loss(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        return loss

    def accuracy_of_each_category(self,y,t):
        y.to_cpu()
        t.to_cpu()
        categories = set(t.data)
        accuracy = {}
        for category in categories:
            supervise_indices = np.where(t.data==category)[0]
            predict_result_of_category = np.argmax(y.data[supervise_indices],axis=1)
            countup = len(np.where(predict_result_of_category==category)[0])
            accuracy[category] = countup
        return accuracy
