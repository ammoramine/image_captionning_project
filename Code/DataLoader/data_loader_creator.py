"""main code to run, all the dependant code is on the repertory "Code"""
# try:
from . import caption_dataset
# except:
    # from Code.DataLoader import caption_dataset
from torch_snippets import *
import pickle

dirFile = os.path.dirname(__file__)


class DataLoaderCreator:
    pathToDataXlsx = os.path.join(dirFile, "../../Data/dataInfo.xlsx")
    assert os.path.exists(pathToDataXlsx)
    path_vocab = os.path.join(dirFile, "../../Code/DataLoader/vocab")
    assert os.path.exists(path_vocab)


    def __init__(self,trainDir=None,valDir=None):
        if (trainDir is None) or (valDir is None):
            self.trainDir = os.path.join(dirFile, "../../Data/train-images")
            self.valDir = os.path.join(dirFile, "../../Data/val-images")
        else:
            self.trainDir,self.valDir = trainDir,valDir
        self.data_info = self.load_data_info()
        self.vocab = self.load_vocabulatory()
        self.trn_ds,self.val_ds = self.construct_data_set()
        self.trn_dl,self.val_dl = self.create_data_loader()
    def load_data_info(self):
        data_info = pd.read_excel(self.pathToDataXlsx)
        return data_info
    #TODO construct vocabulatory if it doesn't exists
    def load_vocabulatory(self):
        # vocab_constructor = vocabulary_constructor.VocabulatoryConstructor(self.data_info)
        # vocab = vocab_constructor.vocab
        vocab = pickle.load(open(self.path_vocab, "rb"))
        return vocab

    def construct_data_set(self ):
        trn_ds = caption_dataset.CaptioningData(self.trainDir, self.data_info[self.data_info['train']], self.vocab)
        val_ds = caption_dataset.CaptioningData(self.valDir, self.data_info[~self.data_info['train']], self.vocab)
        return trn_ds,val_ds

    def create_data_loader(self,batch_size= 32):
        trn_dl = DataLoader(self.trn_ds, batch_size, collate_fn = self.trn_ds.collate_fn)
        val_dl = DataLoader(self.val_ds, batch_size, collate_fn = self.val_ds.collate_fn)
        return trn_dl,val_dl

    def inspect_data_loader(self):
        inspect(*next(iter(self.trn_dl)), names='images,targets,lengths')
    def show_sample_dataset(self):
        image, target, caption = self.trn_ds.choose()
        show(image, title=caption, sz=5)
        print(target)

if __name__ == '__main__':
    import pandas as pd
    trainDir = os.path.join(dirFile, "../../Data/train-images")
    valDir = os.path.join(dirFile,"../../Data/val-images")
    assert os.path.exists(trainDir)
    assert os.path.exists(valDir)

    alg = DataLoaderCreator(trainDir,valDir)

