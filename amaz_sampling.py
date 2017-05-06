import six
import numpy as np

class Sampling(object):


    def __init__(self):
        pass

    def random_sampling(self,epoch,batch_size,data_length):
        """
        yield indices result of random sampling
        """
        for i in six.moves.range(epoch):
            yield  np.random.permutation(data_length)[:batch_size]

    def random_sampling_label_normarize(self,data_length,batch_size,category_num):
        """
        ### FIX ME ###
        yield indices result of random sampling but the sampled-item
        number is equal dependigng on category
        """
        return

    def pick_random_permutation(self,pick_number, sample_number, sort=False):
        pick_number = int(pick_number)
        sample_number = int(sample_number)
        sort = bool(sort)
        if sort:
            return np.sort(np.random.permutation(sample_number)[:pick_number])
        else:
            return np.random.permutation(sample_number)[:pick_number]
