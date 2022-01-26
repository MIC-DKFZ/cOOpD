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

from datamodules.brain import get_brain_dataset

from datamodules.brain import get_brain_dataset, init_arg_dicts, BrainDataLoader, get_brain_dataset_withoutSIMCLR
from config.datasets.brain import get_brain_args


## Readout config values
brain_base_args = get_brain_args(mode='train')
## Args for Datasets used for the contrastive Training (& the generative models)!
datasets_common_args = brain_base_args['common_args']
datasets_train_args = brain_base_args['trainset_args']
d_common_args = dict(**datasets_common_args)

d_train_args = dict(**datasets_train_args)


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



    dir_data = '/home/silvia/Documents/CRADL/pre-processed'

    list_dir = os.listdir(dir_data)

    data = Dataset(dir_data=dir_data,
                   list_IDs=list_dir)
    print(data)
    print(list(range(len(data.list_IDs))))
    print(data.dir_data)
    print(data.list_IDs)

    # Hyperparameters
    batch_size = 3
    num_epochs = 100  # 50
    learning_rate = 1e-6  # 5

    train = DataLoader(dataset= data, batch_size = batch_size, shuffle= True, num_workers= 12)

    train_new = BrainDataLoader(base_dir=[dir_data], list_patients=data.list_IDs, n_items=None)

    anomaly_train, anomaly_val = get_brain_dataset(base_dir=[dir_data],  mode="train", batch_size=20)

    train_loader, val_loader = get_brain_dataset_withoutSIMCLR(base_dir=[dir_data],  mode="train", batch_size=12)
    #print(train_new)


    #dataiter = iter(train)
    #images = dataiter.next()


    #print(images.size())
    #print(images.numpy().shape)



    #
    # for epoch in range(num_epochs):
    #     for b in range(batch_size):
    #         batch = next(train)
    #         print(batch.size())

    # for epoch in range(num_epochs):
    #     loop = tqdm(enumerate(train_new), total=len(train_new), leave=False)
    #     for i, (batch) in loop:
    #         #if epoch == 8:
    #         print('epoch', epoch, i)
    #         print('batch', len(batch['data']))
    #         print('batch', len(batch['input_img']), len(batch['patient_name']))
    #         print('batch', len(batch['data']))
    #         print('batch', (batch['input_img']), (batch['patient_name']))

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(anomaly_train), total=len(anomaly_train), leave=False)
        for i, (batch) in loop:
            # print('epoch', epoch, i)
            # print('batch', len(batch['data']), len(batch['data'][0]), len(batch['data'][1]))
            # print('batch', len(batch['input_img']), len(batch['patient_name']))
            # print('batch', (batch['input_img']), (batch['patient_name']))
            if epoch == 9 and i == 9:
                print('watch')


