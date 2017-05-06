import numpy as np
from chainer import cuda
import chainer

class DataShaping(object):

    def __init__(self,xp):
        self.xp = xp

    def prepareinput(self,data,dtype,volatile=False):
        """
        prepare input data for model
        """
        xp = self.xp
        X = data
        if xp == np:
            inp = np.asarray(X, dtype=dtype)
        else:
            inp = xp.asarray(X, dtype=dtype)
        return chainer.Variable(inp, volatile=volatile)
