from glob import glob
from random import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import trainLoaderConfig, valLoaderConfig
from train import RANDOM_STATE

image_path = r"./dataset/DeepCrack/train_img_aug"
masks_path = r"./dataset/DeepCrack/train_lab_aug"



class CrackData(Dataset):
    def __init__(self, df, img_transforms=None, mask_transform=None, aux_transforms=None):
        self.data = df
        self.img_transform = img_transforms
        self.mask_transform = mask_transform
        self.aux_transforms = aux_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert('RGB')
        mask = Image.open(self.data['masks'].iloc[idx]).convert('L')

        if self.aux_transforms is not None:
            img = self.aux_transforms(img)

        seed = np.random.randint(420)

        random.seed(seed)
        torch.manual_seed(seed)

        img = transforms.functional.equalize(img)
        image = self.img_transform(img)

        random.seed(seed)
        torch.manual_seed(seed)

        mask = self.mask_transform(mask)

        return image, mask


class CrackDataTest(Dataset):
    def __init__(self, df, img_transforms=None, mask_transform=None, aux_transforms=None):
        self.data = df
        self.img_transform = img_transforms
        self.mask_transform = mask_transform
        self.aux_transforms = aux_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert('RGB')
        mask = Image.open(self.data['masks'].iloc[idx]).convert('L')

        if self.aux_transforms is not None:
            img = self.aux_transforms(img)

        seed = np.random.randint(420)

        random.seed(seed)
        torch.manual_seed(seed)

        img = transforms.functional.equalize(img)
        image = self.img_transform(img)

        random.seed(seed)
        torch.manual_seed(seed)

        mask = self.mask_transform(mask)

        return image, mask, self.data['images'].iloc[idx]


def buildDataset(imgs_path, masks_path):
    data = {
        'images': sorted(glob(imgs_path + "/*.jpg")),
        'masks': sorted(glob(masks_path + "/*.png"))
    }
    # test to see if there are images coresponding to masks
    print('数据集大小：' + str(len(data['images'])))
    for img_path, mask_path in zip(data['images'], data['masks']):
        # print(img_path,mask_path)
        assert(img_path[-7:-4] == mask_path[-7:-4])

    df = pd.DataFrame(data)
    dfTrain, dfVal = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    print(len(dfTrain),len(dfVal))
    trainLoader, valLoader = getDataLoaders(dfTrain,
                                            dfVal,
                                            training_data=trainLoaderConfig,
                                            val_data=valLoaderConfig)

    return trainLoader, valLoader

def getDataLoaders(dfTrain, dfVal, **kwargs):
    dataTrain = CrackData(dfTrain,
                          img_transforms=kwargs['training_data']['transforms'],
                          mask_transform=kwargs['training_data']['transforms'],
                          aux_transforms=None)

    trainLoader = DataLoader(dataTrain,
                             batch_size=kwargs['training_data']['batch_size'],
                             shuffle=kwargs['training_data']['shuffle'],
                             pin_memory=torch.cuda.is_available(),
                             num_workers=kwargs['training_data']['num_workers'])

    dataVal = CrackData(dfVal,
                        img_transforms=kwargs['val_data']['transforms'],
                        mask_transform=kwargs['val_data']['transforms'],
                        aux_transforms=None)
    valLoader = DataLoader(dataVal,
                           batch_size=kwargs['val_data']['batch_size'],
                           shuffle=kwargs['val_data']['shuffle'],
                           pin_memory=torch.cuda.is_available(),
                           num_workers=kwargs['val_data']['num_workers'])

    return trainLoader, valLoader


trainLoader, valLoader = buildDataset(image_path, masks_path)