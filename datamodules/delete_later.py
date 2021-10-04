import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor

import os
import numpy as np

import SimpleITK as sitk
import matplotlib.pyplot as plt

from datamodules.extract_patches import extract_patches_3d_fromMask
from batchgenerators.dataloading import MultiThreadedAugmenter

import pandas as pd
from tqdm import tqdm


def imshow(img):
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(npimg)

    plt.show()

class Dataset(Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, dir_data, list_IDs):
        self.dir_data = dir_data
        self.list_IDs = list_IDs

    def __len__(self):
        'denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'Generates one sample of data'
        # considering new_spacing = [0.5, 0.5, 0.5] and that the optimal size for the smallest lung unit (2.5 cm)
        # the dimension of the smallest 3D patch to extract should be 50 voxels = (2.5 cm / 0.05 cm)
        # so, the patch should be 50x50x50

        #select sample
        path_img = os.path.join(self.dir_data, self.list_IDs[item])
        read = np.load(path_img)
        insp = read['insp']
        label = read['label']

        patches = extract_patches_3d_fromMask(insp,label, (1,50,50,50), max_patches=1, random_state=12345)
        print('aqui', patches.shape)
        patches = torch.from_numpy(patches).float()#.long()

        #print(patches)
        #print(patches.size)
        print(patches.shape)

        return patches





if __name__ == "__main__":



    dir_data = '/home/silvia/Documents/CRADL/pre-processed/all'

    list_dir = os.listdir(dir_data)

    data = Dataset(dir_data=dir_data,
                   list_IDs=list_dir)
    print(data)
    print(list(range(len(data.list_IDs))))
    print(data.dir_data)
    print(data.list_IDs)

    # Hyperparameters
    batch_size = 12
    num_epochs = 50  # 50
    learning_rate = 1e-6  # 5

    train = DataLoader(dataset= data, batch_size = batch_size, shuffle= True, num_workers= 12)


    #dataiter = iter(train)
    #images = dataiter.next()


    #print(images.size())
    #print(images.numpy().shape)



    #
    # for epoch in range(num_epochs):
    #     for b in range(batch_size):
    #         batch = next(train)
    #         print(batch.size())



    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train), total=len(train), leave=False)
        for i, (batch) in loop:
            print(batch)
