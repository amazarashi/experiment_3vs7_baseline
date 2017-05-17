import chainer
from chainer import optimizers

class Optimizers(object):

    def __init__(self,model,epoch=300):
        self.model = model
        self.epoch = epoch
        self.optimizer = None

    def __call__(self):
        pass

    def update(self):
        self.optimizer.update()

    def setup(self,model):
        self.optimizer.setup(model)

class OptimizerDarknet(Optimizers):

    def __init__(self,model=None,lr=0.1,momentum=0.9,epoch=160,schedule=(100,160,225),weight_decay=5.0e-4,decay_power=4,batch=64):
        super(OptimizerDarknet,self).__init__(model,epoch)
        self.lr = lr
        self.decay_power = decay_power #polynominal rate decay
        self.optimizer = optimizers.MomentumSGD(self.lr,momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        self.optimizer.setup(model)
        self.optimizer.add_hook(weight_decay)
        self.schedule = schedule
        self.batch = batch
        self.data_length = 50000

    def update_parameter(self,current_epoch):
        # if current_epoch in self.schedule:
        #new_lr = self.lr * (1 - self.batch/self.data_length) ** self.decay_power
        if current_epoch in self.schedule:
            new_lr = self.lr * 0.1
            self.lr = new_lr
            self.optimizer.lr = new_lr
            print("optimizer was changed to {0}..".format(new_lr))


class OptimizerDarknet448(Optimizers):

    def __init__(self,model=None,lr=0.001,momentum=0.9,epoch=160,schedule=(90,145,200),weight_decay=5.0e-4,decay_power=4,batch=64):
        super(OptimizerDarknet,self).__init__(model,epoch)
        self.lr = lr
        self.decay_power = decay_power #polynominal rate decays
        self.optimizer = optimizers.MomentumSGD(self.lr,momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        self.optimizer.setup(model)
        self.optimizer.add_hook(weight_decay)
        self.schedule = schedule
        self.batch = batch
        self.data_length = 50000

    def update_parameter(self,current_epoch):
        # if current_epoch in self.schedule:
        # new_lr = self.lr * (1 - self.batch/self.data_length) ** self.decay_power
        # if current_epoch in self.schedule:
        #     new_lr = new_lr * 0.1
        # self.lr = new_lr
        # self.optimizer.lr = new_lr
        # print("optimizer was changed to {0}..".format(new_lr))
        print("no update")
