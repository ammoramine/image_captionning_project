"""main code to run, all the dependant code is on the repertory "Code"""

from DataLoader import data_loader_creator
import os
import pandas as pd

dirFile = os.path.dirname(__file__)

class MainCode:
    def __init__(self,trainDir,valDir):
        self.trainDir = trainDir
        self.valDir = valDir
        self.data_loader_creator = data_loader_creator.DataLoaderCreator(trainDir, valDir)


if __name__ == '__main__':
    trainDir = os.path.join(dirFile, "../Data/train-images_old")
    valDir = os.path.join(dirFile,"../Data/val-images")

    assert os.path.exists(trainDir)
    assert os.path.exists(valDir)

    alg = MainCode(trainDir,valDir)
