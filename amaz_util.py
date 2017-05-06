import urllib.request
import tarfile
from os import system
import sys
import six
import pickle
from tqdm import tqdm

class Utility(object):

    def __init__(self):
        self.name = "util"

    @staticmethod
    def download_file_fromurl(url,destination,header='--header="Accept: text/html" --user-agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" '):
        cmd = 'wget ' + header + url + ' -O ' + destination + ' -q'
        system(cmd)

    @staticmethod
    def untar_file(filepath):
        if filepath.endswith(".gz"):
            cmd = 'tar -zxvf ' + filepath + ' -C ./'
            system(cmd)
        else:
            print("the given filepath was not tar.gz")

    @staticmethod
    def unpickle(filepath):
        fp = open(filepath, 'rb')
        if sys.version_info.major == 2:
            data = pickle.load(fp)
        elif sys.version_info.major == 3:
            data = pickle.load(fp, encoding='latin-1')
        fp.close()
        return data

    @staticmethod
    def savepickle(data,savepath):
        with open(savepath,'wb') as f:
            pickle.dump(data,f)
        return True

    @staticmethod
    def create_progressbar(end, desc='', stride=1, start=0):
        return tqdm(six.moves.range(int(start), int(end), int(stride)), desc=desc, leave=False)
