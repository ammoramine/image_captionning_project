from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

try:
    from . import vocabulary_constructor
except:
    import vocabulary_constructor
dirFile = os.path.dirname(__file__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CaptioningData(Dataset):
    def __init__(self, root, df, vocab):
        """

        :param root: the path to the images to load
        :param df: the dataFrame that contains informations that links each image to an id
        :param vocab: the vocabulary that converts a word to an integer, identifying uniquely this word
        :param transform: the transform to be applied to each element of the dataset (that consists
        of image and a caption ) , that applies image transformation
        """
        self.df = df.reset_index(drop=True)
        self.root = root
        self.vocab = vocab
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    def tokenize_caption(self,caption):
        """
            coonvert a caption, that consists of a set of word represented as strings
            to a list of tokens (consisting of simple words, or special tokens , that
            can represent for example , the beginning of sentence, end of sentence, or void
            )
        """
        tokens = str(caption).lower().split()
        target = []
        target.append(self.vocab.stoi['<start>'])
        target.extend([self.vocab.stoi[token] for token in tokens])
        target.append(self.vocab.stoi['<end>'])
        target = torch.Tensor(target).long()
        return target
    def  get_image_from_id(self,id):
        """from id of an image, returns the RGB image"""
        image_path = f'{self.root}/{id}.jpg'
        image = Image.open(os.path.join(image_path)).convert('RGB')
        return image
    def __getitem__(self, index):
        """Returns for each index (image target, and caption)."""
        row = self.df.iloc[index].squeeze()
        image = self.get_image_from_id(row.image_id)
        target = self.tokenize_caption(row.caption)
        return image, target, row.caption

    def choose(self):
        """:returns a random element """
        return self[np.random.randint(len(self))]
    def __len__(self):
        return len(self.df)

    def collate_fn(self, data):
        """
            the function takes as input an iterable over samples consisting of an image, a caption
            coded as integers, and the original caption.
            :returns
            images : the images stacked at the order of arrival,
            targets : the numerized target stacked, and padded with 0, until the maximal length of tokens
            length : the number of tokens for each caption, stacked at the same order

            :remarks    function that is returned to the dataloaded, and that is applied to each batch of samples
        """
        data.sort(key=lambda x: len(x[1]), reverse=True) # sort the data, from the image, with the longest caption to the smallest one

        images, targets, captions = zip(*data)
        images = torch.stack([self.transform(image) for image in images], 0)
        lengths = [len(tar) for tar in targets]
        _targets = torch.zeros(len(captions),max(lengths)).long()
        for i, tar in enumerate(targets):
            end = lengths[i]
            _targets[i, :end] = tar[:end]
        return images.to(device), _targets.to(device),torch.tensor(lengths).long().to(device)


if __name__ == '__main__':

    import pandas as pd
    pathToDataXlsx = os.path.join(dirFile, "../Data/dataInfo.xlsx")

    assert os.path.exists(pathToDataXlsx)
    data = pd.read_excel(pathToDataXlsx)
    vocab_constructor = vocabulary_constructor.VocabulatoryConstructor(data)
    vocab = vocab_constructor.vocab

    dirFile = os.path.dirname(__file__)
    trainDir = os.path.join(dirFile, "../Data/train-images_old")
    valDir = os.path.join(dirFile,"../Data/val-images")

    assert os.path.exists(trainDir)
    assert os.path.exists(valDir)

    trn_ds = CaptioningData(trainDir, data[data['train']], vocab)
    val_ds = CaptioningData(valDir, data[~data['train']], vocab)
