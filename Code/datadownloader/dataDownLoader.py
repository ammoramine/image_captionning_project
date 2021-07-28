import os
import pandas as pd
from openimages.download import _download_images_by_id
dirFile = os.path.dirname(__file__)
class DataDownLoader:
    def __init__(self):
        self.pathToDataXlsx = os.path.join(dirFile, "../../Data/dataInfo.xlsx")
        self.pathToTrainImages = os.path.join(dirFile, "../../Data/train-images")
        self.pathToValImages = os.path.join(dirFile, "../../Data/val-images")

        assert os.path.exists(self.pathToDataXlsx)

        self.dataInfo = pd.read_excel(self.pathToDataXlsx,index_col=0)
        self.add_download_collumn_if_necessary()

    def add_download_collumn_if_necessary(self):
        """add downlod collum and set to False if it wasn't present on the data frame"""
        if "downloaded" not in self.dataInfo.columns:
            self.dataInfo["downloaded"] = False
            self.save_data_info()

    def drop_donwloaded_collumn(self):
        """drop downlod"""
        self.dataInfo = self.dataInfo.drop(["downloaded"],axis=1)
        self.save_data_info()

    def reset_download_collumn(self):
        self.drop_donwloaded_collumn()
        self.add_download_collumn_if_necessary()


    def save_data_info(self):
        """sav data info to excel file"""
        self.dataInfo.to_excel(self.pathToDataXlsx)


    def load_train_data(self):
        """download images associated with training data w.r.t to the data_info excel file"""
        subset_imageIds = self.dataInfo[self.dataInfo['train']].image_id.tolist()
        _download_images_by_id(subset_imageIds, 'train', '../Data/train-images/')
    def load_val_data(self):
        """download images associated with testing data w.r.t to the data_info excel file"""
        subset_imageIds = self.dataInfo[~self.dataInfo['train']].image_id.tolist()
        _download_images_by_id(subset_imageIds, 'train', '../Data/val-images/')


    def get_ids_images_downloaded(self):
        full_extensions = os.listdir(self.pathToTrainImages)
        full_extensions + os.listdir(self.pathToValImages)
        ids = [el.split(".jpg")[0] for el in full_extensions]
        return ids
    def add_downloaded_file_to_data_info(self):
        """check in the directory, the image that have been downloaded, and mark them on the excel file"""
        list_ids_imgsdownloaded = self.get_ids_images_downloaded()
        is_img_downloaded = self.dataInfo.image_id.isin(list_ids_imgsdownloaded)
        self.dataInfo.downloaded[is_img_downloaded] = True
if __name__ == '__main__':
    alg = DataDownLoader()