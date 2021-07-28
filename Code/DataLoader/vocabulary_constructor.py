""" module to construct a vocabulatory from a set of captions"""
from torchtext.legacy.data import Field
from pycocotools.coco import COCO
from collections import defaultdict
import pandas as pd
import os,pickle


class VocabulatoryConstructor:
    def __init__(self,data):
        self.data = data
        self.captions_as_txt = data[data['train']]['caption'].tolist()
        self.vocab = self.construct_vocabulatory_from_captions()

    def construct_vocabulatory_from_captions(self):
        captions = Field(sequential=False, init_token='<start>',eos_token='<end>')
        all_tokens = [[w.lower() for w in c.split()] for c in self.captions_as_txt]
        all_tokens = [w for sublist in all_tokens for w in sublist]
        captions.build_vocab(all_tokens)
        return captions.vocab


#TODO: remove that element later, by adding the '<pad>' token, and placing it as output of itos[0]
# and 0 output of unknown character
    def add_pad_tokken_to_at_first_position(self,):
        """
            take vocab element as input and outputs a new custom vocab lightweight element
            containing only the stoi and itos elements, with the '<pad>' element added
        """

        #create a custom vocabulary that contains only the itos and stoi elements
        class Vocab: pass
        vocab = Vocab()
        vocab.itos = self.captions.vocab.itos.copy()
        vocab.itos.insert(0, '<pad>')
        vocab.stoi = defaultdict(lambda: self.captions.vocab.itos.index('<unk>'))
        vocab.stoi['<pad>'] = 0
        for s,i in self.captions.vocab.stoi.items():
            vocab.stoi[s] = i+1
        return vocab


if __name__ == '__main__':
    dirFile = os.path.dirname(__file__)
    pathToDataXlsx = os.path.join(dirFile, "../../Data/dataInfo.xlsx")
    data_info = pd.read_excel(pathToDataXlsx)
    vocab_constructor = VocabulatoryConstructor(data_info)
    vocab = vocab_constructor.vocab
    path_vocab = os.path.join(dirFile,"vocab")
    pickle.dump(vocab, open(path_vocab, "wb"))