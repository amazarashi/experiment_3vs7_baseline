import urllib.request
import tarfile
from os import system
import os
import sys
import cv2
import six
import pickle
from tqdm import tqdm
import numpy as np
import random

import amaz_augumentationPicture
augumentation = amaz_augumentationPicture.AugumentationPicture()

class Normalize32(object):

    @staticmethod
    def train(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(32,36))
        res = augumentation.crop_random(res,(32,32))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.flip_horizontal(res,0.5)
        res = augumentation.convert_to_chainerVariable(res)
        return res

    @staticmethod
    def test(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(32,32))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.convert_to_chainerVariable(res)
        return res

class Normalize64(object):

    @staticmethod
    def train(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(64,68))
        res = augumentation.crop_random(res,(64,64))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.flip_horizontal(res,0.5)
        res = augumentation.convert_to_chainerVariable(res)
        return res

    @staticmethod
    def test(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(64,64))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.convert_to_chainerVariable(res)
        return res

class Normalize128(object):

    @staticmethod
    def train(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(128,132))
        res = augumentation.crop_random(res,(128,128))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.flip_horizontal(res,0.5)
        res = augumentation.convert_to_chainerVariable(res)
        return res

    @staticmethod
    def test(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(128,128))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.convert_to_chainerVariable(res)
        return res

class Normalize224(object):

    @staticmethod
    def train(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(224,230))
        res = augumentation.crop_random(res,(224,224))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.flip_horizontal(res,0.5)
        res = augumentation.convert_to_chainerVariable(res)
        return res

    @staticmethod
    def test(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(224,224))
        #res = augumentation.normalize(res,value=10.)
        res = augumentation.convert_to_chainerVariable(res)
        return res

class Normalize448(object):

    @staticmethod
    def train(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(448,452))
        res = augumentation.crop_random(res,(448,448))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.flip_horizontal(res,0.5)
        res = augumentation.convert_to_chainerVariable(res)
        return res

    @staticmethod
    def test(X):
        res = augumentation.convert_to_imgAry(X)
        res = augumentation.resize(res,(448,448))
        res = augumentation.normalize(res,value=10.)
        res = augumentation.convert_to_chainerVariable(res)
        return res
