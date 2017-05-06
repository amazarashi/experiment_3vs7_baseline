import numpy as np
import amaz_augumentationCustom
import amaz_datashaping
import amaz_augumentation
from chainer import serializers
import pickle

class Tester(object):

    def __init__(self,model=None,dataset=None,dataaugumentation=amaz_augumentationCustom.Normalize128):
        self.name = "tester"
        self.model = model
        self.dataset = dataset
        self.dataaugumentation = dataaugumentation
        self.model_file_path = "model/model_299.pkl"
        self.meta = self.init_meta()
        self.load_model()
        self.xp = np
        self.datashaping = amaz_datashaping.DataShaping(self.xp)

    def init_meta(self):
        meta = self.dataset["meta"]
        return meta

    def load_model(self):
        print("loading model...")
        with open(self.model_file_path, 'rb') as i:
            self.model = pickle.load(i)
        return

    def executeOne(self,x):
        x = amaz_augumentation.Augumentation().Z_score(x)
        da_x = self.dataaugumentation.test(x)
        xin = self.datashaping.prepareinput([da_x],dtype=np.float32,volatile=True)
        y = self.model(xin,train=False)
        res = {}
        score_of_each = list(y.data)
        predict_index = np.argmax(score_of_each, axis=1)
        print(self.meta)
        predict_label = self.meta[predict_index]
        res["score_of_each"] = score_of_each
        res["predict_index"] = int(predict_index)
        res["predict_label"] = predict_label
        return res

# if __name__ == "__main__":
#
